# 🤖 Cross-Platform Sentiment Toward AI  
**A Comparative Analysis of Reddit Tech vs Reddit Non-Tech vs YouTube Tech Communities**

---

## 📖 Table of Contents

1. [Data Collection Overview](#-data-collection-overview)  
2. [Data Preprocessing](#-data-preprocessing)  
3. [Sentiment Distribution](#-sentiment-distribution)  
4. [Confidence Analysis](#-confidence-analysis-bert)  
5. [Chi-Square Tests for Sentiment Distribution](#-chi-square-tests-for-sentiment-distribution)  
6. [Engagement Statistics](#-engagement-statistics)  
7. [Platform Culture Interpretation](#-platform-culture-interpretation)  
8. [Summary](#-summary)  

---

## 🔍 Data Collection Overview

We collected **user-generated content** from a diverse set of Reddit communities and YouTube tech channels, focusing specifically on content that mentioned AI-related keywords.

### 📝 Reddit Sources

- **Non-Technical Subreddits:** Jobs, Education, Journalism, Finance, Music, …  
- **Technical Subreddits:** MachineLearning, ChatGPT, OpenAI, Programming, Computing, …  

### 🎥 YouTube Sources

Collected comments using queries like:  

> "ChatGPT developer tutorial", "AI coding tools", "GitHub Copilot review", "AI replaces programmers"  

**Key Differences**

- **Reddit:** Multiple AI keywords across diverse topics  
- **YouTube:** Mainly “ChatGPT”  

> Platform differences may affect sentiment patterns — YouTube comments cluster around fewer topics and reflect reactions to creators.

**✅ Inclusion Criteria:** Only AI-related content included.

---

## 🧹 Data Preprocessing

We created a **combined dataset (N = 4,178)**:

| Feature | Description |
|---------|-------------|
| 🏷️ category | Reddit Non-Tech, Reddit Tech, YouTube Tech |
| 📝 content | Cleaned text (regex removed markdown, URLs, HTML, whitespace) |
| 👍 score | Engagement (upvotes/likes) |
| ⏰ datetime | Timestamp |
| 🧮 word count | Number of words |
| 📏 content length | Character count |
| 🌐 platform | Reddit or YouTube |
| 😀 sentiment labels | VADER |
| 🤖 sentiment labels & probabilities | BERT |

### 🔧 Sentiment Methods

- **VADER**: Simple, interpretable, limited with sarcasm/tech jargon  
- **BERT**: Handles context, irony, nuanced language — used for analysis  

**BERT Examples**

- **YouTube (Negative, high confidence [0.953]):**  
  "ChatGPT 5 is the biggest piece of garbage of all time…"

- **Reddit Non-Tech (Neutral, high confidence [0.938]):**  
  "ChatGPT was released to the public on November 30, 2022…"

- **Reddit Tech (Positive, high confidence [0.989]):**  
  "Nice. I'm so pumped for the first AI-generated movie…"

---

## 📊 Sentiment Distribution

![Sentiment Distribution](https://github.com/user-attachments/assets/0adc6d0b-d28d-420a-9a05-3a307813ac8b)  

**Key Observations**

Across platforms, the dominant sentiment is neutral. This is expected in AI/tech discussions, which often involve factual updates, feature descriptions, and technical debates. Negative sentiment is moderately high (38%), consistent with skepticism, frustration, and concerns about AI’s risks. Positive sentiment is the smallest group (18%), reflecting fewer “enthusiastic” reactions.

![Platform Comparison](https://github.com/user-attachments/assets/119f1d3a-b24c-45a8-be65-7653747dcb7d)  

- **Reddit Tech:** Most negative (~41%)  
- **Reddit Non-Tech:** Slightly less negative, slightly more positive  
- **YouTube Tech:** Highly positive (~30%)

Reddit Tech is the most negative. Tech enthusiasts are often critical of product performance, ethics, or technical limitations. Reddit Non-Tech is slightly less negative and slightly more positive. Discussions are emotional but more mixed. YouTube Tech is dramatically more positive (30%). This is nearly double the positivity of Reddit.

**Potential Causes for YouTube Positivity**

1. 🎬 Creator influence — video hosts often use an energetic, optimistic tone.
2. 💬 Social norm bias — YouTube comments tend to reward praise toward creators.
3. 🔑 Sampling difference — comments centered around the “ChatGPT” keyword may attract fans and enthusiastic learners.

> Platform culture strongly affects sentiment.

---

## ✅ Confidence Analysis (BERT)

![BERT Confidence](https://github.com/user-attachments/assets/0830f43d-116f-4be6-80ab-583c34a41079)  

- 🔹 ~94% classifications confident (>0.5)  
- 🔹 Positive max confidence = 0.76  
- 🔹 Only 274/4178 samples low confidence → BERT robust

BERT is highly confident overall, with only ~6% uncertain classifications (<0.5). This indicates good signal in the text and consistent differentiation among sentiment classes. The model has a good confident score for almost all data with the highest 0.76 for positive labels. It has low confidence for only a few 274 samples out of 4178 which mean the model handles classification of the data very well.

---

## 📈 Chi-Square Tests for Sentiment Distribution

| Comparison | Chi-square | P-value | Significant? |
|------------|------------|---------|---------------|
| Reddit Tech vs Reddit Non-Tech | 2.505 | 0.2858 | ❌ No |
| Reddit Tech vs YouTube Tech | 91.643 | 0.0000 | ✅ Yes |
| Reddit Non-Tech vs YouTube Tech | 54.334 | 0.0000 | ✅ Yes |

**Interpretation**

This score evaluates whether the proportion of positive, negative, and neutral sentiment is significantly different between categories. The result shows that Reddit tech and non-tech communities are statistically similar with each other while Reddit and Youtube tech difference is extremely large with very strong deviation from expected distribution. Similarly, Reddit non-tech and Youtube tech show strong statistical differences which could not be random. This reinforces that Reddit and YouTube have distinct emotional cultures even on the same topic (AI/ChatGPT).

Chi-square results show that Reddit and YouTube communities form distinct emotional ecosystems.

---

## 📊 Engagement Statistics

![Engagement Overview](https://raw.githubusercontent.com/colaola20/sentiment-analisys-AI/main/READMEimgs/6.1.png)  
![Engagement Screenshot 1](https://github.com/user-attachments/assets/d859c018-633b-49e6-83f7-7a1cb0c02a1f)  
![Engagement Screenshot 2](https://github.com/user-attachments/assets/51a043f6-ce6c-4b82-ad34-68ad20ae821d)  

**Insights**

Looking at Reddit non-tech engagement levels by sentiment, we can see that emotional posts—both positive and negative—tend to go viral, often receiving unusually high numbers of upvotes. In contrast, the pattern in Reddit tech communities is different: the most popular comments are neutral, suggesting that users engage more with factual information, technical news, or announcements. Positive comments also perform well, though not as strongly as neutral ones.

It is also interesting to note that Reddit tech and non-tech subreddits show no significant difference from their sentiment standpoints. The chi-square tests indicate that the proportions of each type of sentiment are relatively the same across both groups, which is interesting because tech and non-tech subreddits can be talking about vastly different things. Tech subreddits contain technical jargon and topics like machine learning as one would expect, but non-tech can span from anything relating to art, education, jobs, etc. A lot of these subreddits share very little in common but it is very interesting to see this because it infers that Reddit’s uniformity for all communities may show a stronger influence and could be considered if other platforms are involved.



### ⚡ ANOVA Results (Engagement by Sentiment)

| Platform | F-statistic | P-value | Significant? |
|----------|------------|---------|---------------|
| Reddit Tech | 1.910 | 0.1484 | ❌ No |
| Reddit Non-Tech | 4.607 | 0.0102 | ✅ Yes |
| YouTube Tech | 0.682 | 0.5059 | ❌ No |

> Engagement influenced by sentiment **only** in Reddit Non-Tech.

YouTube, on the other hand, shows relatively low engagement overall. While positive and negative comments receive more likes than neutral ones, the differences are small. According to the ANOVA results, there is no statistically significant engagement difference across sentiments on either YouTube or Reddit tech, despite the visible numerical differences. However, in Reddit non-tech communities, sentiment does significantly influence engagement, as confirmed by the ANOVA test.

---

## 🌐 Platform Culture Interpretation

The sentiment differences observed across Reddit and YouTube align with well-established patterns in digital sociology and platform behavior research. Studies consistently show that Reddit’s anonymity, threaded discussions, and karma incentives, encourage critical, adversarial, and highly analytical conversation (Chandrasekharan et al., 2020; Massanari, 2015). Users on Reddit, especially within technical communities, frequently challenge claims, critique overhyped technologies, and highlight limitations. This cultural expectation of skepticism maps directly onto our findings: Reddit Tech contains the highest proportion of negative sentiment (41.1%), followed by Reddit Non-Tech (39.1%).

In contrast, YouTube’s social environment is shaped by parasocial dynamics between creators and viewers. Viewers often mirror the tone of content creators, who tend to present AI tools enthusiastically in tutorials, reviews, or demonstrations. Research shows that parasocial closeness reduces critical expression and increases supportive, positive commenting behavior (Chen, 2021; Taecharungroj, 2017). This aligns with our dataset, where YouTube Tech displays nearly double the positivity of Reddit communities (30.1% vs ~16%). Additionally, YouTube’s feedback mechanisms, likes, hearts, and algorithms favoring “agreeable” comments, further suppress negativity and elevate praise.

Together, these platform cultures create distinct emotional ecosystems. Reddit fosters critical evaluation and debate, while YouTube fosters encouragement and affirmational feedback. These sociotechnical differences provide a strong explanatory foundation for the statistically significant sentiment divergences identified in our chi-square tests.

---

## 📌 Summary

- **Data:** 4,178 AI-related posts/comments across Reddit & YouTube  
- **Analysis:** BERT-based sentiment classification  
- **Findings:**  
  - Neutral dominates, negative moderate, positive lowest  
  - Reddit Tech: most negative (~41%)  
  - YouTube Tech: most positive (~30%)  
  - Engagement varies by platform & sentiment  
  - Platform culture drives sentiment differences  

> Reddit & YouTube form **distinct emotional ecosystems**, shaped by platform design and user behavior.

---

✨ **End of Analysis**
