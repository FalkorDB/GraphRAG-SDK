# Contributing to GraphRAG-SDK

Thank you for your interest in contributing to **GraphRAG-SDK**! 
We welcome contributions in any form—whether it's reporting issues, suggesting new features, improving documentation, or submitting code. 
This guide will help you get started.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How to Contribute](#how-to-contribute)
- [Development Setup](#development-setup)
- [Pull Request Guidelines](#pull-request-guidelines)
- [Issue Tracking](#issue-tracking)
- [Testing](#testing)

## Code of Conduct

Please make sure to review and follow our [Code of Conduct](./CODE_OF_CONDUCT.md). 
We expect all contributors to be respectful and considerate in all interactions.

## How to Contribute

### 1. Reporting Bugs

To report a bug:
- First, check the [issue tracker](https://github.com/FalkorDB/GraphRAG-SDK/issues) to see if the issue has already been reported.
- If not, open a new issue with the following information:
  - Steps to reproduce the issue.
  - The expected outcome.
  - The actual outcome, with any relevant error messages or logs.

### 2. Proposing New Features

If you have a feature suggestion:
- Open an issue to start a discussion about the feature and its potential impact.
- Wait for feedback from maintainers before starting any implementation work.

### 3. Improving Documentation

Help us by improving the documentation, which may include:
- Fixing any typos or broken links.
- Adding examples to clarify functionality.
- Submit a pull request directly for minor updates, or open an issue for larger changes.

## Development Setup

To set up **GraphRAG-SDK** for local development:

1. Fork the repository and clone your fork:
   ```bash
   git clone https://github.com/your-username/GraphRAG-SDK.git
   cd GraphRAG-SDK
   ```
   
2. Install the required dependencies:
   ```bash
   poetry install --extras "all"
   ```
   
## Pull Request Guidelines

Before submitting a pull request, please make sure you follow these guidelines:

- **Fork and branch**: Create a new branch from `main` for your work:
   ```bash
   git checkout -b feature-branch
   ```

- **Commits**: Write clear and concise commit messages.
   ```bash
   git commit -m "Add feature X to improve Y"
   ```

- **Link issues**: If your PR addresses an existing issue, mention it in the PR (e.g., `Closes #123`).
- **Testing**: Ensure that your changes are properly tested before submitting the PR.
- **Documentation**: Update any relevant documentation if your changes introduce new features or alter existing behavior.

Once ready, push your branch:
```bash
git push origin feature-branch
```

Then, open a pull request from your fork to the main repository.

## Issue Tracking

We use GitHub Issues for tracking bugs, feature requests, and questions. When opening an issue, please:
- Provide a clear and descriptive title.
- Use labels like **bug**, **enhancement**, or **documentation** to categorize your issue.

## Testing

Testing is essential for ensuring code quality in **GraphRAG-SDK**. Before submitting your changes:
- **Write tests**: Ensure all new functionality is covered by tests.
- **Run tests**: Ensure that all tests pass.
- 
   ```bash
   export PROJECT_ID=${PROJECT_ID}
   export REGION=${REGION}
   export OPENAI_API_KEY=${OPENAI_API_KEY}
   export GOOGLE_API_KEY=${GOOGLE_API_KEY}
   poetry run pytest
   ```

## Thank You!

We truly appreciate your time and effort in contributing to **GraphRAG-SDK**. Every contribution helps improve the project for everyone!
