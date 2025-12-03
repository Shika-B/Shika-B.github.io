+++
title = "Fast and meaningful tokenization for LLMs"
date = "2025-11-29T18:08:29+01:00"

#
# description is optional
#
# description = "An optional description for SEO. If not provided, an automatically created summary will be used."

tags = ["Computer Science", "Machine Learning", "LLMs", "Tokenizers"]
+++

I'm currently taking a [deep learning course](https://jjv.ie/) and I have an assignment that is basically: write a Google Trad/DeepL for French to English translation with minimal data (130k paired sentences) and training time (we need to make it trainable on a laptop). The goal of the assignment is to learn about modern LLMs architectures, specifically about attention and transformers, the deep learning part really. Naturally, this is not what I did: this blog post recounts my wanderings.

From a very high-level perspective, today's LLMs take a context string as an input (the prompt, or the last few prompts and generated answers) and try to predict the next token, i.e. they output a probability distribution over the tokens. Then to generate full sentences as ChatGPT does, or to translate, we can just iterate the predictions:
- Step 1: Break the sentence into words, and break the words into tokens.
- Step 2: If there are $n$ possible token values, the token $i$ is encoded as an $n$-dimensional "one-hot" vector, i.e. a vector with $n$ entries: a $1$ at the $i$-th position and zero elsewhere.
- Step 3:
    - Transformers blocks do their magic (this is not the topic of this post)
    - We get a probability distribution over the tokens that tells us what is the probability that a given token follows after the known context.
    - Pick the most probable token, append it to the context string
    - If that token is not the "shut up" token, we go back to the start of step 3.

Now the transformers part is fun, but everyone is already talking about it. Let's focus on the first part, how do we get tokens ?

## Why bother ?

I am lazy, so I will simply list all words in English, write this:
```python
ENGLISH_WORDS = {'cup': 34874, 'of': 54123, 'tea': 15879, ...}
REVERSE_ENGLISH_WORDS = {34874: 'cup', 54123: 'of', 15879: 'tea', ...}
def tokenize(context):
    return [ENGLISH_WORDS[word] for word in context.split(' ')]

def detokenize(tokens):
    return " ".join(REVERSE_ENGLISH_WORDS[token] for token in tokens)
```

and call it a day? Except for the extra/missing spaces these functions introduce, that is a reasonable option. There a few problems, however:
- We can't have any input that is ill-formed, even a single mistyped word breaks the tokenizing process
- There are a lot of words: almost [a million in English and half a million in French](https://en.wikipedia.org/wiki/List_of_dictionaries_by_number_of_words). That means atleast that many dimensions in our inputs to the model, and that makes for a lot of parameters (I'm training it on a laptop, remember ?)
- The model needs to do everything by itself. Words are often formed by appending meaningful prefixes and suffixes to meaningful roots: take the words low, lowest, lower, hard, hardest and harder. Per word tokenization erases all this information. 
- Some words are very rare, so it is extremely hard to get a dataset where they have enough statistical presence. But most of these words are.. composition of more common roots, prefixes suffixes.
- The preceding point is also true across languages especially when working with Romance languages! Words like "attack" in English and "attaque" in French mean the same thing and have an obvious common root.

In an ideal world, we would thus break sentences into words and words themselves into [lexemes](https://en.wikipedia.org/wiki/Lexeme). It turns out the second task is extremely hard even in reasonable generality, and there is a lot of research going on in the machine learning community to solve it, even among Romance languages.

Conclusion: we need to compromise on our goals. What is the middle ground between splitting on whitespaces and a full linguistics lexeme split, you ask ? A compression algorithm obviously!

## Byte-Pair encoding

Consider the following string `s = "aaabdaaabace"` and imagine a perfect world where all the data is perfectly packed into memory. As the string is formed over an alphabet of 5 tokens, every character can be coded on 3 bits, and we actually get an extra 3 unused token values. Let's use them to compress the string:
```python
VOCAB = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'aaabdaaabace': 5}
```
Easy, right? Yeah, but now our compression scheme is completely unable to generalize to similarly structured but different data. Let's instead enforce that each new token must be the concatenation of two preexisting tokens.

Choosing the extra tokens so that they reduce as much as possible the length of the final string is a hard problem, so let's not solve it exactly and do greedy instead: choose at each step the most common pair of tokens:

```python
s = "aaabdaaabace"  # step 0

"FabdFabace"    # F = aa, occurs 4 times
"FGdFGace"      # G = ab, occurs 2 times
"HdHace"        # H = FG, occurs 2 times 

MERGE_LIST = [(5, (0, 0)), (6, (0, 1)), (7, (5, 6))]
FINAL_VOCAB = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'aa': 5, 'ab': 6, 'aaab': 7}

initial_s = [0, 0, 0, 1, 3, 0, 0, 0, 1, 0, 2, 4]
compressed_s = [7, 3, 7, 9, 2, 4]
```

Now, once we trained our vocab on our initial string, we know `MERGE_LIST` and `FINAL_VOCAB`. To compress any string defined on the same initial alphabet, we go over `MERGE_LIST` and apply merges iteratively. To get our initial string back, we reverse the process.

## Tokenization as compression
In 2016, [Snnrich, Haddow and Birch noticed](https://arxiv.org/abs/1508.07909?utm_source=chatgpt.com) we can actually use this to get a reasonable approximation of the lexeme extraction process. Let's say the dataset contains exactly the following English words:

```python
# POV: training a model on a toaster
ENG_DATASET_WORDS = {'low', 'lower', 'hard', 'harder'}
```

Extracting meaningful lexemes from this dataset amounts to extracting repetitive patterns inside words: 'low', 'er' and 'hard' all occur two times. From that observation, we get a very reasonable token vocabulary of size 3
```python
ENG_VOCAB = {'low': 0, 'hard': 1, 'er': 2}
```

Instead, if we apply the compression algorithm with 6 merges, we get the following merges:
```python
l, o -> lo
e, r -> er
h, a -> ha
r, d -> rd
lo, w -> low
ha, rd -> hard
```
and we get the same final tokens, with some extra intermediate tokens.

## A naive BPE tokenizer 
Let's put that into code. For convenience, we define a token class that stores the token representation, its id and the word it comes from. 
```python
from collections import Counter

class Token:
    def __init__(self, s, tok_id, word_id):
        self.s = s
        self.tok_id = tok_id
        self.word_id = word_id

    def __repr__(self):
        return f"Token({self.s}, tok_id: {self.tok_id} word_id:{self.word_id})"
```

We also define some helper functions respectively building a vocabulary (a `dict[str, int]`) out of a list of words, and counting pairs of token ids coming from the same word 
```python
from collections import Counter

def initial_vocab(words):
    fresh_tok = 0
    vocab = dict()
    for word in words:
        for char in word:
            if not char in vocab:
                vocab[char] = fresh_tok
                fresh_tok += 1
    return vocab

def get_stats(tokens):
    stats = Counter()
    for tok1, tok2 in zip(tokens, tokens[1:]):
        if tok1.word_id == tok2.word_id:
            stats[(tok1.tok_id, tok2.tok_id)] += 1
    return stats
```

Now we need some function that takes a list of tokens and a target pair and merges all consecutive pairs of tokens (inside the same word!) matching the given target pair. The simplest way to do that is to completely rebuild the token list:
```python
def merge(tokens, pair, fresh_token):
    new_tokens = []
    i = 0
    while i < len(tokens) - 1:
        tok1, tok2 = tokens[i], tokens[i+1]
        if tok1.word_id == tok2.word_id and (tok1.tok_id, tok2.tok_id) == pair:
            new_tokens.append(Token(tok1.s + tok2.s, fresh_token, tok1.word_id))
            i += 2
        else:
            new_tokens.append(tok1)
            i += 1
    if i == len(tokens) - 1:
        new_tokens.append(tokens[i])
    return new_tokens
```

Finally, we define a function that trains our tokenizer with a number of merges threshold.
```py
def train(words, num_merges, verbose=True):
    vocab = initial_vocab(words)
    reverse_vocab = {vocab[c]: c for c in vocab.keys()}
    tokens = [Token(c, vocab[c], word_id) for word_id, word in enumerate(words) for c in word]
    
    fresh_token = len(vocab)
    merge_tree = []
    for _ in range(num_merges): 
        stats = get_stats(tokens) 
        try:
            ((left, right), count) = stats.most_common(1)[0]
        except IndexError:
            break
        
        if verbose:
            print(f"Merging the pair ({reverse_vocab[left]}, {reverse_vocab[right]})")
        
        tokens = merge(tokens, (left, right), fresh_token)

        new_token_s = reverse_vocab[left] + reverse_vocab[right]
        vocab[new_token_s] = fresh_token
        reverse_vocab[fresh_token] = new_token_s
        
        merge_tree.append(((left, right), fresh_token))
        fresh_token += 1
    
    return vocab, merge_tree
```

Encoding a list of words is just applying the merges we got from training, and decoding is straightforward
```py
def encode(vocab, merge_tree, words):
    tokens = [Token(c, vocab[c], word_id) for word_id, word in enumerate(words) for c in word]

    for ((left, right), new) in merge_tree:
        tokens = merge(tokens, (left, right), new)
    return tokens

def decode(tokens):
    words = []
    curr_word = []
    i = 0
    while i < len(tokens):
        if i > 0 and tokens[i-1].word_id != tokens[i].word_id:
            words.append("".join(curr_word))
            curr_word = []
        curr_word.append(tokens[i].s)
        i += 1
    words.append("".join(curr_word))    
    return words

words = ['low', 'lower', 'hard', 'harder']

vocab, merge_tree = train(words, 6)
tokens = encode(vocab, merge_tree, words)
print("Tokens: ", [tok.tok_id for tok in tokens])
print("Decoded tokens: ", decode(tokens))
```

```
Merging the pair (l, o)
Merging the pair (lo, w)
Merging the pair (e, r)
Merging the pair (h, a)
Merging the pair (ha, r)
Merging the pair (har, d)
Tokens:  [9, 9, 10, 13, 13, 10]
Decoded tokens:  ['low', 'lower', 'hard', 'harder']
```

Obviously, we cannot use this decoder with a word that contains a 'z' or a '-'. Three options:
- Option 1: Train it on a large enough dataset so that it does not matter
- Option 2: Initialize vocab with the appropriate alphabet for the target language
- Option 2bis: Same as Option 2 but we work on bytes instead of chars, so that we can work with arbitrary unicode strings with an initial token alphabet of 256 bytes. 

The problem with option 2bis is that the model may form token sequences that form invalid unicode characters. Working with unicode strings is painful anyway, so let's say we're working with full ascii strings. The problem with option 2 is that I don't like it because what do you mean I cannot automate the training of the preprocessing for my automated machine translation model? 

Let's do Option 1: [here](https://www.manythings.org/anki/) there is a file called `fra-eng.zip` and after unzipping it we get about 100k pairs of sentences looking like this (not exactly, I removed the extra columns with the credits for the translation to make it fit):
```
You didn't mean it, did you?	Ce n'était pas ton intention, si ?
I don't want to speak ill of the dead.	Je ne veux pas dire du mal des morts.
I would've drowned if you hadn't saved me.	Je me serais noyé si vous ne m'aviez pas sauvé.	
```

We open the file:
```python
with open('data/fra.txt', 'r', encoding='utf-8') as file:
    txt = file.read()
    lines = txt.strip().split('\n')
    fra, eng = [], []
    for line in lines:
        cols = line.split('\t')
        eng.append(cols[0])
        fra.append(cols[1])
print(fra[50], eng[50])
```

```
Hello! Bonjour !
```

Let's split into words (we remove punctuation) and train our BPE on English words:
```python
import re
regex = re.compile("\\s|\\.|\\!|\\?")

eng_words = []
for sentence in eng:
    for word in regex.split(sentence.lower()):
        if len(word) > 0:
            eng_words.append(word)

train(eng_words, 20, verbose=True)
```

```
Merging the pair (t, h)
Merging the pair (o, u)
Merging the pair (t, o)
Merging the pair (i, n)
Merging the pair (r, e)
Merging the pair (y, ou)
Merging the pair (th, e)
Merging the pair (a, t)
Merging the pair (a, n)
Merging the pair (i, s)
Merging the pair (o, n)
Merging the pair (v, e)
Merging the pair (h, e)
Merging the pair (m, e)
Merging the pair (in, g)
Merging the pair (a, s)
Merging the pair (e, d)
Merging the pair (l, l)
Merging the pair (i, t)
Merging the pair (', t)
```
That looks very nice... except these 20 merges took almost a whole minute:
```bash
> time python naive.py
real	0m48,403s
user	0m47,198s
sys	0m0,886s
```

I tried to let it run it's whole course (10k merges) and it took almost 5 hours to train. We need to make things faster if we don't want to leave it running all night long everytime we make a change. We could just Rewrite It In Rust and get insane speed-ups because Python is slow. 
Okay, [challenge accepted](https://github.com/Shika-B/speedy-bpe/tree/main/rust), there we go for 10k merges
```bash
> time cargo run --release
real	5m39,627s
user	4m9,633s
sys	1m26,246s
```
Now it is actually usable, but definitely not fast enough, in real life we do hundreds of thousands of merges and the dataset is magnitudes larger. I feel we can do better, even on a laptop. My Rust code above is clearly not optimized memory-wise, I could avoid quite a few copies by using references over the original string (str slices for those familiar with Rust) instead. Still, I'm using small-strings optimization, and I don't think we could get anything more than a 2x speed-up avoiding these copies and doing a few other small optimizations. It wouldn't get us that last speed-up magnitude we're looking for, so we need to change the actual logic.

If we look at our code, we can sum it up in Pseudo-Python as:
```python
# Say we're training on n words, for m merges
vocab = build_vocab(words) # O(n)
tokens = initial_tokenize(words) # O(n)
merge_tree = []
fresh_id = fresh()
for m in range(m):
    stats = aggregate_stats(tokens) # O(n)
    top_pair, _ = max(stats) # O(n)
    tokens = merge(top_pair, tokens, fresh_id) # O(n)
    merge_tree.append((top_pair, fresh_id)) # O(1)
    fresh_id = fresh() # O(1)
```

which means we're asymptotically doing $mn$ operations. Can we do better ? Yes, we can.

## A faster merge function

Let's go back to the merge function first
```python
def merge(tokens, pair, fresh_token):
    new_tokens = []
    i = 0
    while i < len(tokens) - 1:
        tok1, tok2 = tokens[i], tokens[i+1]
        if tok1.word_id == tok2.word_id and (tok1.tok_id, tok2.tok_id) == pair:
            new_tokens.append(Token(tok1.s + tok2.s, fresh_token, tok1.word_id))
            i += 2
        else:
            new_tokens.append(tok1)
            i += 1
    if i == len(tokens) - 1:
        new_tokens.append(tokens[i])
    return new_tokens
```
What is taking so much time ? We are rebuilding the whole token list even if we are only merging a few pairs among millions of tokens. There are two problems:
- Problem 1: Merging elements inside a list costs a lot, since we need to rebuild the whole list.  
- Problem 2: We do not already know where the tokens we need to merge are.

Problem 1 is solved by consulting the good old cookbook of algorithmics and data structures: we store tokens as a doubly-linked list, and then merging any node with the previous is a piece of cake.

For Problem 2, we could maintain a big map `d: dict[(int, int), list[int]]` such that `d[(tok1.tok_id, tok2.tok_id)]` is a big list containing all the nodes of the linked list such that `node.tok_id = tok1.tok_id` and `node.next.tok_id = tok2.tok_id`. 
Expanding it is easy but maintaining seems hard, because we the values inside the nodes are modified after merges. Lazy is the way: we just check after querying it if the entries are stale by checking if `node.tok_id == tok1.tok_id` and `node.next.tok_id == tok2.tok_id` !

We replace the `Token` class by
```py
class TokenNode:
    def __init__(self, s, tok_id, word_id, nxt = None, prev = None):
        self.s = s
        self.tok_id = tok_id
        self.word_id = word_id
        self.nxt = nxt
        self.prev = prev

    def append_node(self, other):
        self.nxt = other
        other.prev = self
    
    def merge_with_nxt(self, new_tok_id):
        self.s = self.s + self.nxt.s
        self.tok_id = new_tok_id
        self.nxt.tok_id = -1 # Mark it as a dead node, useful later
        self.nxt = self.nxt.nxt
        if self.nxt is not None:
            self.nxt.prev = self
```

In a few places, we have the line
```python
tokens = [Token(c, vocab[c], word_id) for word_id, word in enumerate(words) for c in word]
```
The list comprehension cannot build a linked list, so we replace it by the following function. While we're at it, we build the pairs dictionary
```python
def tokens_and_pairs(words, vocab):
    dummy = TokenNode('', -1, -1)
    node = dummy
    pairs = defaultdict(list)
    for word_id, word in enumerate(words):
        for c in word:
            tok = TokenNode(c, vocab[c], word_id)
            node.append_node(tok)
            if node.word_id == tok.word_id: 
                pairs[(node.tok_id, tok.tok_id)].append(node)
            node = tok
    root = dummy.nxt
    root.prev = None
    return root, pairs
```

and then in encode and train we get
```python
def encode(vocab, merge_tree, words):
    tokens, pairs = tokens_and_pairs(words, vocab) # only change

    for ((left, right), new) in merge_tree:
        merge(pairs, (left, right), new)
    return tokens

def train(words, num_merges, verbose=True):
    vocab = initial_vocab(words)
    reverse_vocab = {vocab[c]: c for c in vocab}
    root, pairs = tokens_and_pairs(words, vocab) # only change

    # ...
```

In decode we replace iterating over tokens by a linked list traversal

```python
def decode(root):
    words = []
    curr_word = []
    node = root
    while node is not None:
        if node.prev is not None and node.prev.word_id != node.word_id:
            print(curr_word)
            words.append("".join(curr_word))
            curr_word = []
        curr_word.append(node.s)
        node = node.nxt
    words.append("".join(curr_word)) 
    return words
```

all of this is just the same logic but . Now for the merge function, we will only iterates over the pairs we care about and check that they are still valid.
```python
# The faster function is actually a little shorter !
def merge(pair_nodes, pair_to_merge, fresh_token):
    for node in pair_nodes[pair_to_merge]:
        if node.nxt is None or (node.tok_id, node.nxt.tok_id) != pair_to_merge:
            continue 
        if node.prev is not None and node.prev.word_id == node.word_id:
            pair_nodes[(node.prev.tok_id, fresh_token)].append(node.prev)
        if node.nxt.nxt is not None and node.nxt.word_id == node.nxt.nxt.word_id:
            pair_nodes[(fresh_token, node.nxt.nxt.tok_id)].append(node)        
        node.merge_with_nxt(fresh_token)
```

Now that our optimization is up and running, let's test it again on 20 merges:
```bash
> python fast.py
real	0m28,841s
user	0m28,149s
sys	0m0,514s
```

Yeeeah... Still going with the Rust code for the moment. We cut off about 10 seconds. Since there's a new overhead at warm-up in the new code, how different is it if we test it with 40 merges:

```bash
40 merges
> time python naive.py
real	1m28,715s
user	1m27,198s
sys	0m1,157s

> time python fast.py
real	0m44,376s
user	0m43,552s
sys	0m0,521s
```

Doubling the amount of merges, we get a 50% increase of compute time with `fast.py` and almost 80% with `naive.py`.

What is the complexity of the new merge function ? It's the number of nodes we had stored in `pair_nodes[pair_to_merge]`, and the total amount of nodes we can merge across all calls is at most the length of the initial token linked list, i.e. n. 

## Updating statistics

Our pseudo-Python code is now
```python
# n words, for m merges
vocab = build_vocab(words) # O(n)
root, pair_nodes = tokens_and_pairs(words) # O(n)
merge_tree = []
fresh_id = fresh()
for m in range(m):
    stats = aggregate_stats(root) # O(n)
    top_pair, _ = max(stats) # O(n)
    tokens = merge(top_pair, root, pair_nodes fresh_id) # O(n) across all calls
    merge_tree.append((top_pair, fresh_id)) # O(1)
    fresh_id = fresh() # O(1)
# Final: O(m * n)
```
The bottleneck is now in these two lines:  
```python
stats = aggregate_stats(root) # O(n)
top_pair, _ = max(stats) # O(n)
```
Instead of rebuilding statistics at each step, we can actually update them after each merge: we just pass it as a parameter of the `merge` function.
```python
def merge(pair_nodes, pair_to_merge, fresh_token, stats=None):
    if stats:
        stats[pair_to_merge] = 0
    for node in pair_nodes[pair_to_merge]:
        if node.nxt is None or (node.tok_id, node.nxt.tok_id) != pair_to_merge:
            continue 
        if node.prev is not None and node.prev.word_id == node.word_id:
            pair_nodes[(node.prev.tok_id, fresh_token)].append(node.prev)
            if stats is not None:
                stats[(node.prev.tok_id, fresh_token)] += 1
                stats[(node.prev.tok_id, node.tok_id)] -= 1
        if node.nxt.nxt is not None and node.nxt.word_id == node.nxt.nxt.word_id:
            pair_nodes[(fresh_token, node.nxt.nxt.tok_id)].append(node)
            if stats is not None:
                stats[(fresh_token, node.nxt.nxt.tok_id)] += 1
                stats[(node.nxt.tok_id, node.nxt.nxt.tok_id)] -= 1
        node.merge_with_nxt(fresh_token)
```
and in the `train` function, we avoid reconstructing the stats dictionary everytime
```python
stats = get_stats(root) # move it out of the loop
for i in range(num_merges): 
```
Note that in the merge function, the stats dictionary may be None because once the tokenizer is trained when we are encoding words, we use the merge function but we do not care about tracking statistics anymore.

How much faster does that get us ?
```bash
> time python fast.py
real	0m33,924s
user	0m33,127s
sys	0m0,627s
```
Nice, we managed to scrap off 10 seconds. Yup, except I lied. I didn't do 40 merges here, I did 10 000 merges this time. You read it right, this is about 10 times faster than the Rust code, in Python. Even on a training set 100 times larger, we would still train our vocabulary in less than an hour, in Python. 

Except it's not good enough, I want real speed:

<div class="tenor-gif-embed" data-postid="14031708" data-share-method="host" data-aspect-ratio="1.9685" data-width="70%"><a href="https://tenor.com/view/speed-i-am-speed-lightning-mcqueen-cars-meme-gif-14031708">Speed I Am Speed GIF</a>from <a href="https://tenor.com/search/speed-gifs">Speed GIFs</a></div> <script type="text/javascript" async src="https://tenor.com/embed.js"></script>

## Actually fixing the complexity

I've been ranting about the complexity of the initial code for quite a while now, but I still made no actual improvement to the overall complexity. Sure, making our implementation more efficient was important, but theoretically nothing changed, we still iterate over the whole dictionary to find the top pair:
```python
# n words, for m merges
vocab = build_vocab(words) # O(n)
root, pair_nodes = tokens_and_pairs(words) # O(n)
merge_tree = []
fresh_id = fresh()
stats = aggregate_stats(root)
for m in range(m):
    top_pair, _ = max(stats) # O(n) !!
    tokens = merge(top_pair, root, pair_nodes fresh_id) # O(n) across all calls
    merge_tree.append((top_pair, fresh_id)) # O(1)
    fresh_id = fresh() # O(1)
# Final complexity is still O(m * n)
```

I think it is time to reopen that old cookbook. We're looking for a data structure that is able to keep track of multiplicity, while giving us access to a fast `max` operation. My personal answer is some kind of bastard between multisets and binary heaps, maintaing a mapping from keys to position in the heap. Double the memory size, but also O(1) max, O(log(n)) insertion/deletion time-wise. We need to be careful during siftdown/up operations because each swap of heap elements needs to be followed by the appropriate update in the dictionary. I'm not going over all the binary heap logic, but if you're interested check [this page](https://runestone.academy/ns/books/published/pythonds/Trees/BinaryHeapImplementation.html) which has crystal clear[^1] explanations.
The code for the `MultisetHeap` class is [here](https://github.com/Shika-B/speedy-bpe/blob/main/python/multiheap.py).

We change our `merge`, `train` and `get_stats` functions to use this new data structure instead of a dictionary.

```python
# In `get_stats`
stats.add((root.tok_id, root.nxt.tok_id), 1) # updated

# In `merge`
stats.add((node.prev.tok_id, fresh_token), 1) # updated
stats.sub((node.prev.tok_id, node.tok_id), 1) # updated
    
stats.add((fresh_token, node.nxt.nxt.tok_id), 1) # updated
stats.sub((node.nxt.tok_id, node.nxt.nxt.tok_id), 1) # updated

# In `train`
(_count, (left, right)) = stats.popmax() # updated
```

Let's try running it... and if you're following along you'll get a weird `KeyError` message. It took me some time to understand this was not a problem in my multiheap implementation but in my logic: When the max is `pop`-ed, `stats[(left, right)]` gives a key error. Now, if `(node.prev.tok_id, node.tok_id) = (left, right)` for instance, boom KeyError. This can happens only if there is a sequence of three characters that are the same, so that's really a PAIN to debug. The fix is, however, very simple:
```python
# In `merge`
stats.add((node.prev.tok_id, fresh_token), 1)
if (node.prev.tok_id, node.tok_id) != pair_to_merge: # updated
    stats.sub((node.prev.tok_id, node.tok_id), 1)
    
stats.add((fresh_token, node.nxt.nxt.tok_id), 1)
if (node.nxt.tok_id, node.nxt.nxt.tok_id) != pair_to_merge: # updated
    stats.sub((node.nxt.tok_id, node.nxt.nxt.tok_id), 1)
```

## Final clean up

While we're at it, let's do a little clean up. Since we're only doing it once, we can build the initial statistics and the linked list in one pass:
```python
def tokens_pairs_and_stats(words, vocab, keep_stats=False):
    dummy = TokenNode("", -1, -1)
    node = dummy
    pairs = defaultdict(list)
    if keep_stats:
        stats = MultisetHeap()

    for word_id, word in enumerate(words):
        for c in word:
            tok = TokenNode(c, vocab[c], word_id)
            node.append_node(tok)
            if node.word_id == tok.word_id:
                pairs[(node.tok_id, tok.tok_id)].append(node)
                if keep_stats:
                    stats.add((node.tok_id, tok.tok_id), 1)
            node = tok
    root = dummy.nxt
    root.prev = None

    if keep_stats:
        return root, pairs, stats
    return root, pairs, None
```
and we replace 
```python
# In train
_root, pairs, stats = tokens_pairs_and_stats(words, vocab, keep_stats=True)

# In encode
tokens, pairs, _ = tokens_pairs_and_stats(words, vocab, keep_stats=False)
```

## Results and conclusion

Running it, we get:

```bash
> time python fast.py
real	0m17,884s
user	0m16,950s
sys	0m0,934s
```

And the final results in a nice little table:
| Code                               |   Time    |
|:---------------------------        |:---------:|
| Python Naive                       | 4h48m     |
| Rust Naive + small strings         | 5m40s     |
| Python Optimized                   | 17s       |
| HuggingFace tokenizers in Rust     | 0.9s      |

The full code is available [here](https://github.com/Shika-B/speedy-bpe/).

I am curious as to how close I can get to hugging face perfs by porting the Python optimized code to Rust. The linked list part is definitely not suited to Rust, but we can [get around that](https://docs.rs/indexlist/latest/indexlist/). Maybe I'll try doing that and update that blog post in the future: stay tuned.



# Comments

Sorry, I'm too lazy to load a proper comment system plugin: see the associated [github issue](https://github.com/Shika-B/Shika-B.github.io/issues/1).

[^1]: except for the fact they use an extra dummy head for a reason I fail to understand (less arithmetic ops ?)