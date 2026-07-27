import sys
sys.path.insert(0, '.')
from astroml.llm.embedding_cache import EmbeddingCache

cache = EmbeddingCache(similarity_threshold=0.85)

# Store 5 entries
texts = [
    'What is the weather today?',
    'How do I detect fraud in transactions?',
    'Explain blockchain technology',
    'Calculate risk score for account',
    'Show me recent transactions',
]
for i, t in enumerate(texts):
    cache.store(t, 'result_' + str(i))

# Exact hits
for t in texts:
    r = cache.get(t)
    assert r is not None, 'exact hit should work for: ' + t

# Semantic hits (slightly varied phrasing)
semantic_queries = [
    'What is the weather outside today?',       # similar to texts[0]
    'How to detect fraudulent transactions?',   # similar to texts[1]
]
for q in semantic_queries:
    r = cache.get(q)
    # Not guaranteed to hit at 0.85 threshold, but log result
    print('semantic query "%s" -> %s' % (q[:40], 'HIT' if r else 'MISS'))

stats = cache.get_stats()
total = stats['hits'] + stats['misses']
print('Hits: %d, Misses: %d, Hit rate: %.2f%%' % (stats['hits'], stats['misses'], stats['hit_rate'] * 100))
print('Sets: %d, Invalidations: %d' % (stats['sets'], stats['invalidations']))
assert stats['hits'] >= 5, 'expected at least 5 exact hits'
print('PASS: exact hits work, hit_rate =', stats['hit_rate'])

# Test invalidation
cache.store('delete me', 'data')
assert cache.get('delete me') == 'data'
removed = cache.invalidate('delete me')
assert removed == True
assert cache.get('delete me') is None
print('PASS: single entry invalidation works')

# Test invalidate_all
count = cache.invalidate_all()
print('PASS: invalidate_all removed %d entries' % count)
after = cache.get_stats()
print('stats after clear:', after)
assert after['sets'] == 0
assert after['hits'] == 0
print('ALL SMOKE TESTS PASSED')
