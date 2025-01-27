// SPDX-License-Identifier: Apache-2.0
// Copyright 2020 - 2022 Pionix GmbH and Contributors to EVerest
#include "OCPP.hpp"

#include <fstream>
#include <optional>
#include <sstream>
#include <string>

#include "generated/types/ocpp.hpp"
#include "ocpp/common/types.hpp"
#include "ocpp/v16/types.hpp"
#include <fmt/core.h>
#include <ocpp_conversions.hpp>

#include <conversions.hpp>
#include <error_mapping.hpp>
#include <evse_security_ocpp.hpp>
#include <external_energy_limits.hpp>

namespace module {

const std::string CERTS_SUB_DIR = "certs";
const std::string SQL_CORE_MIGRTATIONS = "core_migrations";
const std::string INOPERATIVE_ERROR_TYPE = "evse_manager/Inoperative";
const std::string SWITCHING_PHASES_REASON = "SwitchingPhases";

namespace fs = std::filesystem;

/// \brief Converts the given \p error into the ErrorInfo that contains all necessary data for a
/// StatusNotification.req
static ocpp::v16::ErrorInfo get_error_info(const Everest::error::Error& error) {

    const auto error_type = error.type;
    const auto uuid = error.uuid.uuid;

    auto it = std::find_if(MREC_ERROR_MAP.begin(), MREC_ERROR_MAP.end(),
                           [&](const auto& entry) { return error_type.find(entry.first) != std::string::npos; });

    // is MREC error
    if (it != MREC_ERROR_MAP.end()) {
        // lambda to create MREC error info
        auto make_mrec_error_info = [&](ocpp::v16::ChargePointErrorCode code, const std::string& vendor_error_code) {
            return ocpp::v16::ErrorInfo{uuid, code, false, std::nullopt, CHARGE_X_MREC_VENDOR_ID, vendor_error_code};
        };
        return make_mrec_error_info(it->second.first, it->second.second);
    }

    if (error_type == INOPERATIVE_ERROR_TYPE) {
        return ocpp::v16::ErrorInfo{uuid,      ocpp::v16::ChargePointErrorCode::OtherError,
                                    true,      "EVSE is inoperative",
                                    "EVerest", "caused_by:" + error.message};
    }

    const auto get_simplified_error_type = [](const std::string& error_type) {
        // this function should return everything after the first '/'
        // delimiter - if there is no delimiter or the delimiter is at
        // the end, it should return the input itself
        static constexpr auto TYPE_INTERFACE_DELIMITER = '/';

        auto input = std::istringstream(error_type);
        std::string tmp;

        // move right after the first delimiter
        std::getline(input, tmp, TYPE_INTERFACE_DELIMITER);

        if (!input) {
            // no delimiter found or delimiter at the end
            return error_type;
        }

        // get the rest of the input
        std::getline(input, tmp);

        return tmp;
    };

    const auto is_fault = [](const Everest::error::Error&) {
        // NOTE (aw): this could be customized, depending on the error
        return false;
    };

    static constexpr auto TYPE_DELIMITER = '/';

    return {
        uuid,
        ocpp::v16::ChargePointErrorCode::OtherError,
        is_fault(error),
        error.origin.to_string(),                                                // info
        error.message,                                                           // vendor id
        get_simplified_error_type(error.type) + TYPE_DELIMITER + error.sub_type, // vendor error code
    };
}

void create_empty_user_config(const fs::path& user_config_path) {
    if (fs::exists(user_config_path.parent_path())) {
        std::ofstream fs(user_config_path.c_str());
        auto user_config = json::object();
        fs << user_config << std::endl;
        fs.close();
    } else {
        EVLOG_AND_THROW(
            std::runtime_error(fmt::format("Provided UserConfigPath is invalid: {}", user_config_path.string())));
    }
}

void OCPP::set_external_limits(const std::map<int32_t, ocpp::v16::EnhancedChargingSchedule>& charging_schedules) {
    const auto start_time = ocpp::DateTime();

    // iterate over all schedules reported by the libocpp to create ExternalLimits
    // for each connector
    for (auto const& [connector_id, schedule] : charging_schedules) {

        if (not external_energy_limits::is_evse_sink_configured(this->r_evse_energy_sink, connector_id)) {
            EVLOG_warning << "Can not apply external limits! No evse energy sink configured for evse_id: "
                          << connector_id;
            continue;
        }

        types::energy::ExternalLimits limits;
        std::vector<types::energy::ScheduleReqEntry> schedule_import;
        for (const auto period : schedule.chargingSchedulePeriod) {
            types::energy::ScheduleReqEntry schedule_req_entry;
            types::energy::LimitsReq limits_req;
            const auto timestamp = start_time.to_time_point() + std::chrono::seconds(period.startPeriod);
            schedule_req_entry.timestamp = ocpp::DateTime(timestamp).to_rfc3339();
            if (period.numberPhases.has_value()) {
                limits_req.ac_max_phase_count = period.numberPhases.value();
            }
            if (schedule.chargingRateUnit == ocpp::v16::ChargingRateUnit::A) {
                limits_req.ac_max_current_A = period.limit;
                if (schedule.minChargingRate.has_value()) {
                    limits_req.ac_min_current_A = schedule.minChargingRate.value();
                }
            } else {
                limits_req.total_power_W = period.limit;
            }
            schedule_req_entry.limits_to_leaves = limits_req;
            schedule_import.push_back(schedule_req_entry);
        }
        limits.schedule_import.emplace(schedule_import);
        auto& evse_sink = external_energy_limits::get_evse_sink_by_evse_id(this->r_evse_energy_sink, connector_id);
        evse_sink.call_set_external_limits(limits);
    }
}

void OCPP::publish_charging_schedules(
    const std::map<int32_t, ocpp::v16::EnhancedChargingSchedule>& charging_schedules) {
    // publish the schedule over mqtt
    types::ocpp::ChargingSchedules schedules;
    for (const auto& charging_schedule : charging_schedules) {
        types::ocpp::ChargingSchedule sch = conversions::to_charging_schedule(charging_schedule.second);
        sch.evse = charging_schedule.first;
        schedules.schedules.emplace_back(std::move(sch));
    }
    this->p_ocpp_generic->publish_charging_schedules(schedules);
}

void OCPP::process_session_event(int32_t evse_id, const types::evse_manager::SessionEvent& session_event) {
    auto everest_connector_id = session_event.connector_id.value_or(1);
    auto ocpp_connector_id = this->evse_connector_map[evse_id][everest_connector_id];

    if (session_event.event == types::evse_manager::SessionEventEnum::Enabled) {
        this->charge_point->on_enabled(evse_id);
    } else if (session_event.event == types::evse_manager::SessionEventEnum::Disabled) {
        EVLOG_debug << "EVSE#" << evse_id << ": "
                    << "Received Disabled";
        this->charge_point->on_disabled(evse_id);
    } else if (session_event.event == types::evse_manager::SessionEventEnum::TransactionStarted) {
        EVLOG_info << "EVSE#" << evse_id << ": "
                   << "Received TransactionStarted";
        const auto transaction_started = session_event.transaction_started.value();

        const auto timestamp = ocpp_conversions::to_ocpp_datetime_or_now(session_event.timestamp);
        const auto energy_Wh_import = transaction_started.meter_value.energy_Wh_import.total;
        const auto session_id = session_event.uuid;
        const auto id_token = transaction_started.id_tag.id_token.value;
        const auto signed_meter_value = transaction_started.signed_meter_value;
        std::optional<int32_t> reservation_id_opt = std::nullopt;
        if (transaction_started.reservation_id) {
            reservation_id_opt.emplace(transaction_started.reservation_id.value());
        }
        std::optional<std::string> signed_meter_data;
        if (signed_meter_value.has_value()) {
            // there is no specified way of transmitting signing method, encoding method and public key
            // this has to be negotiated beforehand or done in a custom data transfer
            signed_meter_data.emplace(signed_meter_value.value().signed_meter_data);
        }
        this->charge_point->on_transaction_started(ocpp_connector_id, session_event.uuid, id_token, energy_Wh_import,
                                                   reservation_id_opt, timestamp, signed_meter_data);
    } else if (session_event.event == types::evse_manager::SessionEventEnum::ChargingPausedEV) {
        EVLOG_debug << "Connector#" << ocpp_connector_id << ": "
                    << "Received ChargingPausedEV";
        this->charge_point->on_suspend_charging_ev(ocpp_connector_id);
    } else if (session_event.event == types::evse_manager::SessionEventEnum::ChargingPausedEVSE or
               session_event.event == types::evse_manager::SessionEventEnum::WaitingForEnergy) {
        EVLOG_debug << "Connector#" << ocpp_connector_id << ": "
                    << "Received ChargingPausedEVSE";
        this->charge_point->on_suspend_charging_evse(ocpp_connector_id);
    } else if (session_event.event == types::evse_manager::SessionEventEnum::SwitchingPhases) {
        EVLOG_debug << "Connector#" << ocpp_connector_id << ": "
                    << "Received SwitchingPhases";
        this->charge_point->on_suspend_charging_evse(ocpp_connector_id, SWITCHING_PHASES_REASON);
    } else if (session_event.event == types::evse_manager::SessionEventEnum::ChargingStarted ||
               session_event.event == types::evse_manager::SessionEventEnum::ChargingResumed) {
        EVLOG_debug << "Connector#" << ocpp_connector_id << ": "
                    << "Received ChargingResumed";
        this->charge_point->on_resume_charging(ocpp_connector_id);
    } else if (session_event.event == types::evse_manager::SessionEventEnum::TransactionFinished) {
        EVLOG_debug << "Connector#" << ocpp_connector_id << ": "
                    << "Received TransactionFinished";

        const auto transaction_finished = session_event.transaction_finished.value();
        const auto timestamp = ocpp_conversions::to_ocpp_datetime_or_now(session_event.timestamp);
        const auto energy_Wh_import = transaction_finished.meter_value.energy_Wh_import.total;
        const auto signed_meter_value = transaction_finished.signed_meter_value;

        auto reason = ocpp::v16::Reason::Other;
        if (transaction_finished.reason.has_value()) {
            reason = conversions::to_ocpp_reason(transaction_finished.reason.value());
        }
        std::optional<ocpp::CiString<20>> id_tag_opt = std::nullopt;
        if (transaction_finished.id_tag.has_value()) {
            id_tag_opt.emplace(ocpp::CiString<20>(transaction_finished.id_tag.value().id_token.value));
        }
        std::optional<std::string> signed_meter_data;
        if (signed_meter_value.has_value()) {
            // there is no specified way of transmitting signing method, encoding method and public key
            // this has to be negotiated beforehand or done in a custom data transfer
            signed_meter_data.emplace(signed_meter_value.value().signed_meter_data);
        }
        this->charge_point->on_transaction_stopped(ocpp_connector_id, session_event.uuid, reason, timestamp,
                                                   energy_Wh_import, id_tag_opt, signed_meter_data);
        // always triggered by libocpp
    } else if (session_event.event == types::evse_manager::SessionEventEnum::SessionStarted) {
        EVLOG_info << "Connector#" << ocpp_connector_id << ": "
                   << "Received SessionStarted";
        // ev side disconnect
        auto session_started = session_event.session_started.value();
        this->charge_point->on_session_started(ocpp_connector_id, session_event.uuid,
                                               conversions::to_ocpp_session_started_reason(session_started.reason),
                                               session_started.logging_path);
    } else if (session_event.event == types::evse_manager::SessionEventEnum::SessionFinished) {
        EVLOG_debug << "Connector#" << ocpp_connector_id << ": "
                    << "Received SessionFinished";
        // ev side disconnect
        this->evse_soc_map[evse_id].reset();
        this->charge_point->on_session_stopped(ocpp_connector_id, session_event.uuid);
    } else if (session_event.event == types::evse_manager::SessionEventEnum::ReservationStart) {
        this->charge_point->on_reservation_start(ocpp_connector_id);
    } else if (session_event.event == types::evse_manager::SessionEventEnum::ReservationEnd) {
        this->charge_point->on_reservation_end(ocpp_connector_id);
    } else if (session_event.event == types::evse_manager::SessionEventEnum::PluginTimeout) {
        this->charge_point->on_plugin_timeout(ocpp_connector_id);
    }
}

void OCPP::init_evse_subscriptions() {
    int32_t evse_id = 1;
    for (auto& evse : this->r_evse_manager) {
        evse->subscribe_powermeter([this, evse_id](types::powermeter::Powermeter powermeter) {
            ocpp::Measurement measurement;
            measurement.power_meter = conversions::to_ocpp_power_meter(powermeter);
            if (this->evse_soc_map[evse_id].has_value()) {
                // soc is present, so add this to the measurement
                measurement.soc_Percent = ocpp::StateOfCharge{this->evse_soc_map[evse_id].value()};
            }
            if (powermeter.temperatures.has_value()) {
                measurement.temperature_C = conversions::to_ocpp_temperatures(powermeter.temperatures.value());
            }
            this->charge_point->on_meter_values(evse_id, measurement);
        });

        evse->subscribe_ev_info([this, evse_id](const types::evse_manager::EVInfo& ev_info) {
            if (ev_info.soc.has_value()) {
                this->evse_soc_map[evse_id] = ev_info.soc.value();
            }
        });

        evse->subscribe_limits([this, evse_id](types::evse_manager::Limits limits) {
            double max_current = limits.max_current;
            this->charge_point->on_max_current_offered(evse_id, max_current);
        });

        evse->subscribe_session_event([this, evse_id](types::evse_manager::SessionEvent session_event) {
            if (this->ocpp_stopped) {
                // dont call any on handler in case ocpp is stopped
                return;
            }

            if (session_event.event == types::evse_manager::SessionEventEnum::SessionResumed) {
                this->resuming_session_ids.insert(session_event.uuid);
                return;
            }

            if (!this->started) {
                EVLOG_info << "OCPP not fully initialized, but received a session event on evse_id: " << evse_id
                           << " that will be queued up: " << session_event.event;
                std::scoped_lock lock(this->session_event_mutex);
                this->event_queue[evse_id].push(session_event);
                return;
            }

            this->process_session_event(evse_id, session_event);
        });

        evse->subscribe_iso15118_certificate_request(
            [this, evse_id](types::iso15118_charger::RequestExiStreamSchema request) {
                this->charge_point->data_transfer_pnc_get_15118_ev_certificate(
                    evse_id, request.exi_request, request.iso15118_schema_version,
                    conversions::to_ocpp_certificate_action_enum(request.certificate_action));
            });

        evse_id++;
    }
}

void OCPP::init_evse_connector_map() {
    int32_t ocpp_connector_id = 1; // this represents the OCPP connector id
    int32_t evse_id = 1;           // this represents the evse id of EVerests evse manager
    for (const auto& evse : this->r_evse_manager) {
        const auto _evse = evse->call_get_evse();
        std::map<int32_t, int32_t> connector_map; // maps EVerest connector_id to OCPP connector_id

        if (_evse.id != evse_id) {
            throw std::runtime_error("Configured evse_id(s) must be starting with 1 counting upwards");
        }
        for (const auto& connector : _evse.connectors) {
            connector_map[connector.id] = ocpp_connector_id;
            this->connector_evse_index_map[ocpp_connector_id] =
                evse_id - 1; // - 1 to specify the index for r_evse_manager
            ocpp_connector_id++;
        }

        if (connector_map.size() == 0) {
            this->connector_evse_index_map[ocpp_connector_id] =
                evse_id - 1; // - 1 to specify the index for r_evse_manager
            connector_map[1] = ocpp_connector_id;
            ocpp_connector_id++;
        }

        this->evse_connector_map[_evse.id] = connector_map;
        evse_id++;
    }
}

void OCPP::init_evse_maps() {
    std::lock_guard<std::mutex> lk(this->evse_ready_mutex);
    for (size_t evse_id = 1; evse_id <= this->r_evse_manager.size(); evse_id++) {
        this->evse_ready_map[evse_id] = false;
        this->evse_soc_map[evse_id] = std::nullopt;
    }
}

bool OCPP::all_evse_ready() {
    for (auto const& [evse, ready] : this->evse_ready_map) {
        if (!ready) {
            return false;
        }
    }
    EVLOG_info << "All EVSE ready. Starting OCPP1.6 service";
    return true;
}

ocpp::v16::ChargingRateUnit get_unit_or_default(const std::string& unit_string) {
    try {
        return ocpp::v16::conversions::string_to_charging_rate_unit(unit_string);
    } catch (const std::out_of_range& e) {
        EVLOG_warning << "RequestCompositeScheduleUnit configured incorrectly with: " << unit_string
                      << ". Defaulting to using Amps.";
        return ocpp::v16::ChargingRateUnit::A;
    }
}

void OCPP::init() {
    invoke_init(*p_main);
    invoke_init(*p_ocpp_generic);
    invoke_init(*p_auth_validator);
    invoke_init(*p_auth_provider);
    invoke_init(*p_data_transfer);

    // ensure all evse_energy_sink(s) that are connected have an evse id mapping
    for (const auto& evse_sink : this->r_evse_energy_sink) {
        if (not evse_sink->get_mapping().has_value()) {
            EVLOG_critical << "Please configure an evse mapping your configuration file for the connected "
                              "r_evse_energy_sink with module_id: "
                           << evse_sink->module_id;
            throw std::runtime_error("At least one connected evse_energy_sink misses a mapping to an evse.");
        }
    }

    const auto error_handler = [this](const Everest::error::Error& error) {
        const auto evse_id = error.origin.mapping.has_value() ? error.origin.mapping.value().evse : 0;
        const auto error_info = get_error_info(error);
        if (this->started) {
            this->charge_point->on_error(evse_id, error_info);
        } else {
            this->event_queue[evse_id].push(error_info);
        }
    };

    const auto error_cleared_handler = [this](const Everest::error::Error& error) {
        const auto evse_id = error.origin.mapping.has_value() ? error.origin.mapping.value().evse : 0;

        if (this->started) {
            this->charge_point->on_error_cleared(evse_id, error.uuid.uuid);
        } else {
            this->event_queue[evse_id].push(error.uuid.uuid);
        }
    };

    subscribe_global_all_errors(error_handler, error_cleared_handler);

    this->init_evse_maps();

    for (size_t evse_id = 1; evse_id <= this->r_evse_manager.size(); evse_id++) {
        this->r_evse_manager.at(evse_id - 1)->subscribe_waiting_for_external_ready([this, evse_id](bool ready) {
            std::lock_guard<std::mutex> lk(this->evse_ready_mutex);
            if (ready) {
                this->evse_ready_map[evse_id] = true;
                this->evse_ready_cv.notify_one();
            }
        });

        // also use the the ready signal, TODO(kai): maybe warn about it's usage here`
        this->r_evse_manager.at(evse_id - 1)->subscribe_ready([this, evse_id](bool ready) {
            std::lock_guard<std::mutex> lk(this->evse_ready_mutex);
            if (ready) {
                if (!this->evse_ready_map[evse_id]) {
                    EVLOG_error << "Received EVSE ready without receiving waiting_for_external_ready first, this is "
                                   "probably a bug in your evse_manager implementation / configuration. evse_id: "
                                << evse_id;
                }
                this->evse_ready_map[evse_id] = true;
                this->evse_ready_cv.notify_one();
            }
        });
    }

    this->ocpp_share_path = this->info.paths.share;

    auto configured_config_path = fs::path(this->config.ChargePointConfigPath);

    // try to find the config file if it has been provided as a relative path
    if (!fs::exists(configured_config_path) && configured_config_path.is_relative()) {
        configured_config_path = this->ocpp_share_path / configured_config_path;
    }
    if (!fs::exists(configured_config_path)) {
        EVLOG_AND_THROW(Everest::EverestConfigError(
            fmt::format("OCPP config file is not available at given path: {} which was "
                        "resolved to: {}",
                        this->config.ChargePointConfigPath, configured_config_path.string())));
    }
    const auto config_path = configured_config_path;
    EVLOG_info << "OCPP config: " << config_path.string();

    auto configured_user_config_path = fs::path(this->config.UserConfigPath);
    // try to find the user config file if it has been provided as a relative path
    if (!fs::exists(configured_user_config_path) && configured_user_config_path.is_relative()) {
        configured_user_config_path = this->ocpp_share_path / configured_user_config_path;
    }
    const auto user_config_path = configured_user_config_path;
    EVLOG_info << "OCPP user config: " << user_config_path.string();

    std::ifstream ifs(config_path.c_str());
    std::string config_file((std::istreambuf_iterator<char>(ifs)), (std::istreambuf_iterator<char>()));
    auto json_config = json::parse(config_file);
    json_config.at("Core").at("NumberOfConnectors") = this->r_evse_manager.size();

    if (fs::exists(user_config_path)) {
        std::ifstream ifs(user_config_path.c_str());
        std::string user_config_file((std::istreambuf_iterator<char>(ifs)), (std::istreambuf_iterator<char>()));

        try {
            const auto user_config = json::parse(user_config_file);
            EVLOG_info << "Augmenting chargepoint config with user_config entries";
            json_config.merge_patch(user_config);
        } catch (const json::parse_error& e) {
            EVLOG_error << "Error while parsing user config file.";
            EVLOG_AND_THROW(e);
        }
    } else {
        EVLOG_debug << "No user-config provided. Creating user config file";
        create_empty_user_config(user_config_path);
    }

    const auto sql_init_path = this->ocpp_share_path / SQL_CORE_MIGRTATIONS;
    if (!fs::exists(this->config.MessageLogPath)) {
        try {
            fs::create_directory(this->config.MessageLogPath);
        } catch (const fs::filesystem_error& e) {
            EVLOG_AND_THROW(e);
        }
    }

    this->charge_point = std::make_unique<ocpp::v16::ChargePoint>(
        json_config.dump(), this->ocpp_share_path, user_config_path, std::filesystem::path(this->config.DatabasePath),
        sql_init_path, std::filesystem::path(this->config.MessageLogPath),
        std::make_shared<EvseSecurity>(*this->r_security));

    this->charge_point->set_message_queue_resume_delay(std::chrono::seconds(config.MessageQueueResumeDelay));

    this->init_evse_subscriptions(); // initialize EvseManager subscriptions as early as possible
}

void OCPP::ready() {
    invoke_ready(*p_main);
    invoke_ready(*p_ocpp_generic);
    invoke_ready(*p_auth_validator);
    invoke_ready(*p_auth_provider);
    invoke_ready(*p_data_transfer);

    this->init_evse_connector_map();

    this->charge_point->register_pause_charging_callback([this](int32_t connector) {
        if (this->connector_evse_index_map.count(connector)) {
            return this->r_evse_manager.at(this->connector_evse_index_map.at(connector))->call_pause_charging();
        } else {
            return false;
        }
    });
    this->charge_point->register_resume_charging_callback([this](int32_t connector) {
        if (this->connector_evse_index_map.count(connector)) {
            return this->r_evse_manager.at(this->connector_evse_index_map.at(connector))->call_resume_charging();
        } else {
            return false;
        }
    });
    this->charge_point->register_stop_transaction_callback([this](int32_t connector, ocpp::v16::Reason reason) {
        if (this->connector_evse_index_map.count(connector)) {
            types::evse_manager::StopTransactionRequest req;
            req.reason = conversions::to_everest_stop_transaction_reason(reason);
            return this->r_evse_manager.at(this->connector_evse_index_map.at(connector))->call_stop_transaction(req);
        } else {
            return false;
        }
    });

    this->charge_point->register_unlock_connector_callback([this](int32_t connector) {
        if (this->connector_evse_index_map.count(connector)) {
            EVLOG_info << "Executing unlock connector callback";
            return this->r_evse_manager.at(this->connector_evse_index_map.at(connector))->call_force_unlock(1);
        } else {
            return false;
        }
    });

    // int32_t reservation_id, CiString<20> auth_token, DateTime expiry_time,
    // std::string parent_id
    this->charge_point->register_reserve_now_callback([this](int32_t reservation_id, int32_t connector,
                                                             ocpp::DateTime expiryDate, ocpp::CiString<20> idTag,
                                                             std::optional<ocpp::CiString<20>> parent_id) {
        types::reservation::Reservation reservation;
        reservation.id_token = idTag.get();
        reservation.reservation_id = reservation_id;
        reservation.expiry_time = expiryDate.to_rfc3339();

        if (parent_id) {
            reservation.parent_id_token.emplace(parent_id.value().get());
        }

        if (connector == 0) {
            reservation.evse_id = std::nullopt;
        } else {
            reservation.evse_id = connector;
        }

        auto response = this->r_reservation->call_reserve_now(reservation);
        return conversions::to_ocpp_reservation_status(response);
    });

    this->charge_point->register_upload_diagnostics_callback([this](const ocpp::v16::GetDiagnosticsRequest& msg) {
        types::system::UploadLogsRequest upload_logs_request;
        upload_logs_request.location = msg.location;

        if (msg.stopTime.has_value()) {
            upload_logs_request.latest_timestamp.emplace(msg.stopTime.value().to_rfc3339());
        }
        if (msg.startTime.has_value()) {
            upload_logs_request.oldest_timestamp.emplace(msg.startTime.value().to_rfc3339());
        }
        if (msg.retries.has_value()) {
            upload_logs_request.retries.emplace(msg.retries.value());
        }
        if (msg.retryInterval.has_value()) {
            upload_logs_request.retry_interval_s.emplace(msg.retryInterval.value());
        }
        const auto upload_logs_response = this->r_system->call_upload_logs(upload_logs_request);
        ocpp::v16::GetLogResponse response;
        if (upload_logs_response.file_name.has_value()) {
            response.filename.emplace(ocpp::CiString<255>(upload_logs_response.file_name.value()));
        }
        response.status = conversions::to_ocpp_log_status_enum_type(upload_logs_response.upload_logs_status);
        return response;
    });

    this->charge_point->register_upload_logs_callback([this](ocpp::v16::GetLogRequest msg) {
        types::system::UploadLogsRequest upload_logs_request;
        upload_logs_request.location = msg.log.remoteLocation;
        upload_logs_request.request_id = msg.requestId;
        upload_logs_request.type = ocpp::v16::conversions::log_enum_type_to_string(msg.logType);

        if (msg.log.latestTimestamp.has_value()) {
            upload_logs_request.latest_timestamp.emplace(msg.log.latestTimestamp.value().to_rfc3339());
        }
        if (msg.log.oldestTimestamp.has_value()) {
            upload_logs_request.oldest_timestamp.emplace(msg.log.oldestTimestamp.value().to_rfc3339());
        }
        if (msg.retries.has_value()) {
            upload_logs_request.retries.emplace(msg.retries.value());
        }
        if (msg.retryInterval.has_value()) {
            upload_logs_request.retry_interval_s.emplace(msg.retryInterval.value());
        }

        const auto upload_logs_response = this->r_system->call_upload_logs(upload_logs_request);

        ocpp::v16::GetLogResponse response;
        if (upload_logs_response.file_name.has_value()) {
            response.filename.emplace(ocpp::CiString<255>(upload_logs_response.file_name.value()));
        }
        response.status = conversions::to_ocpp_log_status_enum_type(upload_logs_response.upload_logs_status);
        return response;
    });
    this->charge_point->register_update_firmware_callback([this](const ocpp::v16::UpdateFirmwareRequest msg) {
        types::system::FirmwareUpdateRequest firmware_update_request;
        firmware_update_request.location = msg.location;
        firmware_update_request.request_id = -1;
        firmware_update_request.retrieve_timestamp.emplace(msg.retrieveDate.to_rfc3339());
        if (msg.retries.has_value()) {
            firmware_update_request.retries.emplace(msg.retries.value());
        }
        if (msg.retryInterval.has_value()) {
            firmware_update_request.retry_interval_s.emplace(msg.retryInterval.value());
        }
        this->r_system->call_update_firmware(firmware_update_request);
    });

    this->charge_point->register_signed_update_firmware_callback([this](ocpp::v16::SignedUpdateFirmwareRequest msg) {
        types::system::FirmwareUpdateRequest firmware_update_request;
        firmware_update_request.request_id = msg.requestId;
        firmware_update_request.location = msg.firmware.location;
        firmware_update_request.signature.emplace(msg.firmware.signature.get());
        firmware_update_request.signing_certificate.emplace(msg.firmware.signingCertificate.get());
        firmware_update_request.retrieve_timestamp.emplace(msg.firmware.retrieveDateTime.to_rfc3339());

        if (msg.firmware.installDateTime.has_value()) {
            firmware_update_request.install_timestamp.emplace(msg.firmware.installDateTime.value());
        }
        if (msg.retries.has_value()) {
            firmware_update_request.retries.emplace(msg.retries.value());
        }
        if (msg.retryInterval.has_value()) {
            firmware_update_request.retry_interval_s.emplace(msg.retryInterval.value());
        }

        const auto system_response = this->r_system->call_update_firmware(firmware_update_request);

        return conversions::to_ocpp_update_firmware_status_enum_type(system_response);
    });

    this->charge_point->register_all_connectors_unavailable_callback([this]() {
        EVLOG_info << "All connectors unavailable, proceed with firmware installation";
        this->r_system->call_allow_firmware_installation();
    });

    // FIXME(kai): subscriptions should be as early as possible
    this->r_system->subscribe_log_status([this](types::system::LogStatus log_status) {
        this->charge_point->on_log_status_notification(log_status.request_id,
                                                       types::system::log_status_enum_to_string(log_status.log_status));
    });

    this->r_system->subscribe_firmware_update_status(
        [this](types::system::FirmwareUpdateStatus firmware_update_status) {
            this->charge_point->on_firmware_update_status_notification(
                firmware_update_status.request_id,
                conversions::to_ocpp_firmware_status_notification(firmware_update_status.firmware_update_status));
        });

    this->charge_point->register_provide_token_callback(
        [this](const std::string& id_token, std::vector<int32_t> referenced_connectors, bool prevalidated) {
            types::authorization::ProvidedIdToken provided_token;
            provided_token.id_token = {id_token, types::authorization::IdTokenType::Central};
            provided_token.authorization_type = types::authorization::AuthorizationType::OCPP;
            provided_token.connectors.emplace(referenced_connectors);
            provided_token.prevalidated.emplace(prevalidated);
            this->p_auth_provider->publish_provided_token(provided_token);
        });

    this->charge_point->register_set_connection_timeout_callback(
        [this](int32_t connection_timeout) { this->r_auth->call_set_connection_timeout(connection_timeout); });

    this->charge_point->register_disable_evse_callback([this](int32_t connector) {
        if (this->connector_evse_index_map.count(connector)) {
            return this->r_evse_manager.at(this->connector_evse_index_map.at(connector))
                ->call_enable_disable(
                    0, {types::evse_manager::Enable_source::CSMS, types::evse_manager::Enable_state::Disable, 5000});
        } else {
            return false;
        }
    });

    this->charge_point->register_set_system_time_callback(
        [this](const std::string& system_time) { this->r_system->call_set_system_time(system_time); });

    this->charge_point->register_enable_evse_callback([this](int32_t connector) {
        if (this->connector_evse_index_map.count(connector)) {
            return this->r_evse_manager.at(this->connector_evse_index_map.at(connector))
                ->call_enable_disable(
                    0, {types::evse_manager::Enable_source::CSMS, types::evse_manager::Enable_state::Enable, 5000});
        } else {
            return false;
        }
    });

    this->charge_point->register_cancel_reservation_callback(
        [this](int32_t reservation_id) { return this->r_reservation->call_cancel_reservation(reservation_id); });

    if (this->config.EnableExternalWebsocketControl) {
        const std::string connect_topic = "everest_api/ocpp/cmd/connect";
        this->mqtt.subscribe(connect_topic,
                             [this](const std::string& data) { this->charge_point->connect_websocket(); });

        const std::string disconnect_topic = "everest_api/ocpp/cmd/disconnect";
        this->mqtt.subscribe(disconnect_topic,
                             [this](const std::string& data) { this->charge_point->disconnect_websocket(); });
    }

    this->charge_point->register_is_token_reserved_for_connector_callback(
        [this](const int32_t connector, const std::string& id_token) -> ocpp::ReservationCheckStatus {
            types::reservation::ReservationCheck reservation_check_request;
            reservation_check_request.evse_id = connector;
            reservation_check_request.id_token = id_token;

            types::reservation::ReservationCheckStatus status =
                this->r_reservation->call_exists_reservation(reservation_check_request);

            return ocpp_conversions::to_ocpp_reservation_check_status(status);
        });

    const auto composite_schedule_unit = get_unit_or_default(this->config.RequestCompositeScheduleUnit);

    // publish charging schedules at least once on startup
    const auto charging_schedules = this->charge_point->get_all_enhanced_composite_charging_schedules(
        this->config.PublishChargingScheduleDurationS, composite_schedule_unit);
    this->set_external_limits(charging_schedules);
    this->publish_charging_schedules(charging_schedules);

    this->charging_schedules_timer = std::make_unique<Everest::SteadyTimer>([this, composite_schedule_unit]() {
        const auto charging_schedules = this->charge_point->get_all_enhanced_composite_charging_schedules(
            this->config.PublishChargingScheduleDurationS, composite_schedule_unit);
        this->set_external_limits(charging_schedules);
        this->publish_charging_schedules(charging_schedules);
    });
    if (this->config.PublishChargingScheduleIntervalS > 0) {
        this->charging_schedules_timer->interval(std::chrono::seconds(this->config.PublishChargingScheduleIntervalS));
    }

    this->charge_point->register_signal_set_charging_profiles_callback([this, composite_schedule_unit]() {
        // this is executed when CSMS sends new ChargingProfile that is accepted by
        // the ChargePoint
        EVLOG_info << "Received new Charging Schedules from CSMS";
        const auto charging_schedules = this->charge_point->get_all_enhanced_composite_charging_schedules(
            this->config.PublishChargingScheduleDurationS, composite_schedule_unit);
        this->set_external_limits(charging_schedules);
        this->publish_charging_schedules(charging_schedules);
    });

    this->charge_point->register_is_reset_allowed_callback([this](ocpp::v16::ResetType type) {
        const auto reset_type = conversions::to_everest_reset_type(type);
        return this->r_system->call_is_reset_allowed(reset_type);
    });

    this->charge_point->register_reset_callback([this](ocpp::v16::ResetType type) {
        const auto reset_type = conversions::to_everest_reset_type(type);
        this->r_system->call_reset(reset_type, false);
    });

    this->charge_point->register_connection_state_changed_callback(
        [this](bool is_connected) { this->p_ocpp_generic->publish_is_connected(is_connected); });

    this->charge_point->register_get_15118_ev_certificate_response_callback(
        [this](const int32_t connector_id, const ocpp::v201::Get15118EVCertificateResponse& certificate_response,
               const ocpp::v201::CertificateActionEnum& certificate_action) {
            types::iso15118_charger::ResponseExiStreamStatus response;
            response.status = conversions::to_everest_iso15118_charger_status(certificate_response.status);
            response.certificate_action = conversions::to_everest_certificate_action_enum(certificate_action);
            if (not certificate_response.exiResponse.get().empty()) {
                // since exi_response is an optional in the EVerest type we only set it when not empty
                response.exi_response.emplace(certificate_response.exiResponse.get());
            }

            this->r_evse_manager.at(this->connector_evse_index_map.at(connector_id))
                ->call_set_get_certificate_response(response);
        });

    this->charge_point->register_security_event_callback([this](const std::string& type, const std::string& tech_info) {
        EVLOG_info << "Security Event in OCPP occured: " << type;
        types::ocpp::SecurityEvent event;
        event.type = type;
        event.info = tech_info;
        this->p_ocpp_generic->publish_security_event(event);
    });

    this->charge_point->register_transaction_started_callback(
        [this](const int32_t connector, const std::string& session_id) {
            types::ocpp::OcppTransactionEvent tevent;
            tevent.transaction_event = types::ocpp::TransactionEvent::Started;
            tevent.evse = {connector, 1};
            tevent.session_id = session_id;
            p_ocpp_generic->publish_ocpp_transaction_event(tevent);
        });

    this->charge_point->register_transaction_updated_callback(
        [this](const int32_t connector, const std::string& session_id, const int32_t transaction_id,
               const ocpp::v16::IdTagInfo& id_tag_info) {
            types::ocpp::OcppTransactionEvent tevent;
            tevent.transaction_event = types::ocpp::TransactionEvent::Updated;
            tevent.evse = {connector, 1};
            tevent.session_id = session_id;
            tevent.transaction_id = std::to_string(transaction_id);
            p_ocpp_generic->publish_ocpp_transaction_event(tevent);
        });

    this->charge_point->register_transaction_stopped_callback(
        [this](const int32_t connector, const std::string& session_id, const int32_t transaction_id) {
            EVLOG_info << "Transaction stopped at connector: " << connector << "session_id: " << session_id;
            types::ocpp::OcppTransactionEvent tevent;
            tevent.transaction_event = types::ocpp::TransactionEvent::Ended;
            tevent.evse = {connector, 1};
            tevent.session_id = session_id;
            tevent.transaction_id = std::to_string(transaction_id);
            p_ocpp_generic->publish_ocpp_transaction_event(tevent);
        });

    this->charge_point->register_boot_notification_response_callback(
        [this](const ocpp::v16::BootNotificationResponse& boot_notification_response) {
            const auto everest_boot_notification_response =
                conversions::to_everest_boot_notification_response(boot_notification_response);
            this->p_ocpp_generic->publish_boot_notification_response(everest_boot_notification_response);
        });

    this->charge_point->register_session_cost_callback(
        [this](const ocpp::RunningCost& session_cost,
               const uint32_t number_of_decimals) -> ocpp::v16::DataTransferResponse {
            const types::session_cost::SessionCost cost =
                ocpp_conversions::create_session_cost(session_cost, number_of_decimals, {});
            ocpp::v16::DataTransferResponse response;
            this->p_session_cost->publish_session_cost(cost);
            response.status = ocpp::v16::DataTransferStatus::Accepted;
            return response;
        });

    this->charge_point->register_set_display_message_callback(
        [this](const std::vector<ocpp::DisplayMessage>& messages) -> ocpp::v16::DataTransferResponse {
            ocpp::v16::DataTransferResponse response;
            if (this->r_display_message.empty()) {
                EVLOG_warning << "No display message handler registered, dropping data transfer display message";
                response.status = ocpp::v16::DataTransferStatus::Rejected;
                return response;
            }
            std::vector<types::display_message::DisplayMessage> display_messages;
            for (const ocpp::DisplayMessage& message : messages) {
                const types::display_message::DisplayMessage m = ocpp_conversions::to_everest_display_message(message);
                display_messages.push_back(m);
            }

            const types::display_message::SetDisplayMessageResponse display_message_response =
                this->r_display_message.at(0)->call_set_display_message(display_messages);
            response = conversions::to_ocpp_data_transfer_response(display_message_response);

            return response;
        });

    if (!this->r_data_transfer.empty()) {
        this->charge_point->register_data_transfer_callback([this](const ocpp::v16::DataTransferRequest& request) {
            types::ocpp::DataTransferRequest data_transfer_request;
            data_transfer_request.vendor_id = request.vendorId.get();
            if (request.messageId.has_value()) {
                data_transfer_request.message_id = request.messageId.value().get();
            }
            data_transfer_request.data = request.data;
            types::ocpp::DataTransferResponse data_transfer_response =
                this->r_data_transfer.at(0)->call_data_transfer(data_transfer_request);
            ocpp::v16::DataTransferResponse response;
            response.status = conversions::to_ocpp_data_transfer_status(data_transfer_response.status);
            response.data = data_transfer_response.data;
            return response;
        });
    }

    std::unique_lock lk(this->evse_ready_mutex);
    while (!this->all_evse_ready()) {
        this->evse_ready_cv.wait(lk);
    }

    const auto boot_reason = conversions::to_ocpp_boot_reason_enum(this->r_system->call_get_boot_reason());
    if (this->charge_point->start({}, boot_reason, this->resuming_session_ids)) {
        // signal that we're started
        this->started = true;
        EVLOG_info << "OCPP initialized";

        // process session event queue
        std::scoped_lock lock(this->session_event_mutex);
        for (auto& [evse_id, evse_event_queue] : this->event_queue) {
            while (!evse_event_queue.empty()) {
                auto queued_event = evse_event_queue.front();
                if (std::holds_alternative<types::evse_manager::SessionEvent>(queued_event)) {
                    const auto session_event = std::get<types::evse_manager::SessionEvent>(queued_event);
                    EVLOG_info << "Processing queued event for evse_id: " << evse_id
                               << ", event: " << session_event.event;
                    this->process_session_event(evse_id, session_event);
                } else if (std::holds_alternative<ocpp::v16::ErrorInfo>(queued_event)) {
                    const auto error = std::get<ocpp::v16::ErrorInfo>(queued_event);
                    EVLOG_info << "Processing queued error event for evse_id: " << evse_id
                               << ", error id: " << error.uuid;
                    this->charge_point->on_error(evse_id, error);
                } else {
                    // holds string -> is event to clear error
                    const auto cleared_uuid = std::get<ClearedErrorId>(queued_event);
                    EVLOG_info << "Processing queued cleared error event for evse_id: " << evse_id
                               << ", error id: " << cleared_uuid;
                    this->charge_point->on_error_cleared(evse_id, cleared_uuid);
                }
                evse_event_queue.pop();
            }
        }
    }

    // signal to the EVSEs that OCPP is initialized
    for (const auto& evse : this->r_evse_manager) {
        evse->call_external_ready_to_start_charging();
    }
}

int32_t OCPP::get_ocpp_connector_id(int32_t evse_id, int32_t connector_id) {
    return this->evse_connector_map.at(evse_id).at(connector_id);
}

} // namespace module
