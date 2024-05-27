## Usage

### Manual schema creation

```python
s = Schema()
s.add_entity('Actor').add_attribute('name', str, unique=True)
s.add_entity('Movie').add_attribute('title', str, unique=True)
s.add_relation("ACTED", 'Actor', 'Movie')

g = KG("IMDB", schema=s)
g.add_source("./data/madoff.txt")

g.create()

answer, messages = g.ask("List a few actors")
```

### Automatic schema creation

```python
sources = [Source("./data/madoff.txt")]
s = Schema.auto_detect(sources)

g = KG("IMDB", schema=s)
g.add_source("./data/madoff.txt")

g.create()

answer, messages = g.ask("List a few actors")
```
