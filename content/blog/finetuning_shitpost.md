+++
title = "LLMs are a strong baseline for shitposting"
date = "2026-04-18T6:08:29+01:00"

#
# description is optional
#
# description = "An optional description for SEO. If not provided, an automatically created summary will be used."

tags = ["Computer Science", "Machine Learning", "LLMs", "Bot"]
+++

Last weekend I was talking with some friends online, as I do most of my weekends. I asked one of them to elaborate on something, he got lazy and jokingly answered with an obvious ChatGPT answer. Someone else wondered if we couldn't just plug a GPT to the discord server, as a bot for instance. I said I had a better idea. A few sleep-deprived days later, we had a new friend, [Zizou](https://wikipedia.com/wiki/Zinedine_Zidane), spitting our brand of nonsense. Here's the whole story.

# The data

The first step was getting the data. I used [DiscordChatExporter](https://github.com/tyrrrz/discordchatexporter), a scraper in Python that crawls through servers, and exported about 1 million messages. That's a lot, but most of it is noise. Single word messages, links to YouTube videos nobody watched, random GIFs, etc. I didn't want to train on all of that, I wanted the model to learn the *good* stuff.

So I filtered by reactions. Any message that received at least N reactions from the group (I tuned this threshold until the dataset felt right, about 10k messages made the cut) got labeled a "banger" and kept. Pretty much crowdsourced quality filtering[^1].

I want the bot to be able to "complete" the context with a joke, so the training setup is fairly simple. I feed it the last few messages before the banger and it tries to predict the joke. 

# Working around my hardware: (Q)LoRA

Before I get into the training saga, a quick note on LoRA (Low-Rank Adaptation), since I'll be mentioning it a lot.

Finetuning a model the traditional way means updating all its weights, as you would do when first training it. Basically, for a 3B parameter model, that's 3 billion 16-bit floats (= 6 billion bytes ~ 6 GB) you need to store, and then all the gradients that come with them. I am doing all this on my laptop GPU, a RTX 2050 with 4GB of VRAM: that's a hard no.
 
LoRA works around this with a clever approximation: instead of updating the weight matrix $W$ directly, you freeze it and add a low-rank perturbation $\Delta W = AB$ where $A \in \mathbb{R}^{d \times r}$ and $B \in \mathbb{R}^{r \times k}$, with rank $r \ll \min(d, k)$, and you only train $A$ and $B$.

The bet is that you don't need to touch the whole weight space to specialize a model; a small low-rank update is enough. I tried $r = 8$ and $r = 16$, which is less than 1% of the original parameters.

If you're following, you noticed that 6GB is still above the 4GB I have: to load a 3B model, LoRA or not, I still need to load it into GPU memory to run the inference. I could load it one layer at a time, but that would make everything magnitudes slower: remember, I only have the weekend. That's where my next trick comes in:

QLoRA (Quantized LoRA) pushes the above idea further. As the remaining VRAM cost in plain LoRA is the frozen base model itself, QLoRA quantizes those frozen weights to 4 bits, which cuts the base model's footprint by roughly 4x. A 7B model that would eat ~14GB at fp16 fits in ~4GB quantized. The LoRA adapters themselves are kept as floating points and trained normally, only the frozen part is quantized. At inference, weights are dequantized on the fly[^2] just before each matrix multiplication, so the computation itself still happens in a reasonable precision.

The accuracy loss from all this is supposedly small, and in our case it will be, but in general it is non-zero[^3].

# The training saga

Models under a certain size are basically useless for this. I tried GPT-2 Medium (0.35B), Qwen 2.5 0.5B, 1.5B, and then 3B with QLoRA. Same result every time: bits of rhythm, occasional vocabulary, context basically gone. Ask a question and 80% of the time the reply wasn’t even remotely related. The 3B was marginally better; it would at least stay on topic, sometimes. It was also right at the edge of what my 4GB card could handle, and very slow to train. I accepted I wasn’t going to get what I wanted out of my hardware.

I probably would’ve stopped there, except out of roughly 500 messages, maybe five actually got it right. Those rare moments where the model clearly understood the context, the jokes were actually funny.

## Qwen3 8B on RunPod — this is the one

I spent about $30 on a RunPod instance with a proper GPU and finetuned Qwen3 8B-Instruct with no quantization, full LoRA. It's just better. The model actually gets context. It calls back to something said three messages ago, responds to the vibe of a message rather than just its literal content, and sometimes produces something that genuinely sounds like it came from one of us.

Sadly, hosting an LLM is costly, and once there was nothing left of the $30 to host it, we had to kill Zizou.

# What I'd do differently

The reaction threshold worked okay as a quality filter, but it's pretty crude. It probably throws out dry humor and niche references that only get one or two reactions. Weighting by reaction count and using a softer cutoff would probably do better.

But for now, the bot occasionally says something funny enough that people forget I extracted all our private conversations to feed them into an LLM. That's good enough.

# Comments

Sorry, I'm too lazy to load a proper comment system plugin: see the associated [github issue](https://github.com/Shika-B/Shika-B.github.io/issues/3).

[^1]: No underpaid third-worlders were involved! 

[^2]: The specific format used for quantization is NF4 (NormalFloat 4-bit), introduced in the QLoRA paper. It exploits the fact that neural network weights are approximately normally distributed (most values cluster near 0, with exponentially fewer values in the tails). Standard quantization is uniform, which wastes most of its precision on regions where almost no weights live. NF4 instead places its 16 levels at the **quantiles** of a standard normal distribution. The only catch is that GPUs have no native NF4 arithmetic, so weights are dequantized to f16 on the fly before each matrix multiply, adding a small compute overhead.

[^3]: When Anthropic quantizes the models I am paying a subscription for, it's not okay. When I do it, it's fine. 

 