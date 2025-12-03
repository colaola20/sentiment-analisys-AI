## **Cross-Platform Sentiment Toward AI: A Comparison of Reddit Tech vs Reddit non-Tech vs Youtube Tech communities**

### 1. Data collection overview

We collected user-generated content from a diverse set of Reddit communities and YouTube tech channels, focusing specifically on content that mentioned AI-related keywords.

### Reddit Sources
- Non-Technical Subreddits
  - Jobs
  - Education
  - Journalism
  - Finance
  - Music
- Technical Subreddits
  - MachineLearning
  - ChatGPT
  - OpenAI
  - Programming
  - Computing

### Youtube Sources
Comments were collected from videos using search queries such as “ChatGPT developer tutorial”, “AI coding tools”, “GitHub Copilot review”, “AI replaces programmers”, etc.

### Important Differences
- Reddit - multiple keywords
- YouTube - mainly restricted to “chatgpt”

This likely affects sentiment patterns because YouTube comments cluster around fewer topics and may reflect reactions to specific creators.

### 2. Data Preprocessing
We built a combined dataset (N = 4178) containing:

- category (Reddit Non-Tech, Reddit Tech, YouTube Tech)
- content (cleaned using regex for markdown, URLs, HTML, whitespace)
- score (engagement: upvotes or likes)
- datetime
- word count, content length
- platform
- sentiment labels via VADER
- sentiment labels and probability scores via BERT

The dataset contains no missing values.

Two sentiment methods were compared:

- VADER – captures clear sentiment but weak with sarcasm, tech jargon
- BERT – handles context, irony, and nuanced language
  
We use BERT only for the final analysis due to superior performance on social-media style text.

Example BERT classifications confirm correct behavior:
- Negative (strong insult)
- Neutral (factual statement)
- Positive (enthusiastic/expressive)


### Some Examples of BERT Scoring

Youtube

NEGATIVE (high confidence):

[0.953] ChatGPT 5 is the biggest piece of garbage of all time…

Reddit non-tech

NEUTRAL (high confidence):

[0.938] ChatGPT was released to the public on November 30, 2022…

Reddit tech

POSITIVE (high confidence):

[0.989] Nice. I'm so pumped for the first ai generated movie…

### 3. Sentiment Distribution

<img src="https://raw.githubusercontent.com/colaola20/sentiment-analisys-AI/main/READMEimgs/3.1.png" width="500">

Across platforms, the dominant sentiment is neutral. This is expected in AI/tech discussions, which often involve factual updates, feature descriptions, and technical debates. Negative sentiment is moderately high (38%), consistent with skepticism, frustration, and concerns about AI’s risks. Positive sentiment is the smallest group (18%), reflecting fewer “enthusiastic” reactions.

<img src="https://raw.githubusercontent.com/colaola20/sentiment-analisys-AI/main/READMEimgs/3.2.png" width="1000">

Reddit Tech is the most negative. Tech enthusiasts are often critical of product performance, ethics, or technical limitations. Reddit Non-Tech is slightly less negative and slightly more positive. Discussions are emotional but more mixed. YouTube Tech is dramatically more positive (30%). This is nearly double the positivity of Reddit.

Possible causes:

1. Creator influence — video hosts often use an energetic, optimistic tone.
2. Social norm bias — YouTube comments tend to reward praise toward creators.
3. Sampling difference — comments centered around the “ChatGPT” keyword may attract fans and enthusiastic learners.

This indicates platform culture differences strongly impact sentiment.

### 4. Confidence Analysis

<img src="https://raw.githubusercontent.com/colaola20/sentiment-analisys-AI/main/READMEimgs/4.1.png" width="500">

<img src="https://raw.githubusercontent.com/colaola20/sentiment-analisys-AI/main/READMEimgs/4.2.png" width="1000">

BERT is highly confident overall, with only ~6% uncertain classifications (<0.5). This indicates good signal in the text and consistent differentiation among sentiment classes. 
The model has a good confident score for almost all data with the highest 0.76 for positive labels. It has low confidence for only a few 274 samples out of 4178 which mean the model handles classification of the data very well.


### 5. Chi-Square Tests for Sentiment Distribution

Reddit Tech vs Reddit Non-Tech:
- Chi-square: 2.505
- P-value: 0.2858
- Significant: NO

Reddit Tech vs YouTube Tech:
- Chi-square: 91.643
- P-value: 0.0000
- Significant: YES

Reddit Non-Tech vs YouTube Tech:
- Chi-square: 54.334
- P-value: 0.0000
- Significant: YES

This score evaluates whether the proportion of positive, negative, and neutral sentiment is significantly different between categories. The result shows that Reddit tech and non-tech communities are statistically similar with each other while Reddit and Youtube tech difference is extremely large with very strong deviation from expected distribution. Similarly, Reddit non-tech and Youtube tech show strong statistical differences which could not be random. This reinforces that Reddit and YouTube have distinct emotional cultures even on the same topic (AI/ChatGPT).

Chi-square results show that Reddit and YouTube communities form distinct emotional ecosystems.


### 6. Engagement Statistics

<img src="https://raw.githubusercontent.com/colaola20/sentiment-analisys-AI/main/READMEimgs/6.1.png" width="750">

<img src="https://raw.githubusercontent.com/colaola20/sentiment-analisys-AI/main/READMEimgs/6.2.png" width="1000">

<img src="https://raw.githubusercontent.com/colaola20/sentiment-analisys-AI/main/READMEimgs/6.3.png" width="500"> <img src="https://raw.githubusercontent.com/colaola20/sentiment-analisys-AI/main/READMEimgs/6.4.png" width="500">

<img src="https://raw.githubusercontent.com/colaola20/sentiment-analisys-AI/main/READMEimgs/6.5.png" width="1000">

Looking at Reddit non-tech engagement levels by sentiment, we can see that emotional posts—both positive and negative—tend to go viral, often receiving unusually high numbers of upvotes. In contrast, the pattern in Reddit tech communities is different: the most popular comments are neutral, suggesting that users engage more with factual information, technical news, or announcements. Positive comments also perform well, though not as strongly as neutral ones.

It is also interesting to note that Reddit tech and non-tech subreddits show no significant difference from their sentiment standpoints. The chi-square tests indicate that the proportions of each type of sentiment are relatively the same across both groups, which is interesting because tech and non-tech subreddits can be talking about vastly different things. Tech subreddits contain technical jargon and topics like machine learning as one would expect, but non-tech can span from anything relating to art, education, jobs, etc. A lot of these subreddits share very little in common but it is very interesting to see this because it infers that Reddit’s uniformity for all communities may show a stronger influence and could be considered if other platforms are involved.

Engagement Differences (ANOVA):

Reddit Tech:
- F-statistic: 1.910
- P-value: 0.1484
- Significant: NO

Reddit Non-Tech:
- F-statistic: 4.607
- P-value: 0.0102
- Significant: YES


YouTube Tech:
- F-statistic: 0.682
- P-value: 0.5059
- Significant: NO

YouTube, on the other hand, shows relatively low engagement overall. While positive and negative comments receive more likes than neutral ones, the differences are small. According to the ANOVA results, there is no statistically significant engagement difference across sentiments on either YouTube or Reddit tech, despite the visible numerical differences. However, in Reddit non-tech communities, sentiment does significantly influence engagement, as confirmed by the ANOVA test.


### Platform Culture Interpretation

The sentiment differences observed across Reddit and YouTube align with well-established patterns in digital sociology and platform behavior research.
Studies consistently show that Reddit’s anonymity, threaded discussions, and karma incentives, encourage critical, adversarial, and highly analytical conversation (Chandrasekharan et al., 2020; Massanari, 2015). Users on Reddit, especially within technical communities, frequently challenge claims, critique overhyped technologies, and highlight limitations. This cultural expectation of skepticism maps directly onto our findings: Reddit Tech contains the highest proportion of negative sentiment (41.1%), followed by Reddit Non-Tech (39.1%).

In contrast, YouTube’s social environment is shaped by parasocial dynamics between creators and viewers. Viewers often mirror the tone of content creators, who tend to present AI tools enthusiastically in tutorials, reviews, or demonstrations. Research shows that parasocial closeness reduces critical expression and increases supportive, positive commenting behavior (Chen, 2021; Taecharungroj, 2017). This aligns with our dataset, where YouTube Tech displays nearly double the positivity of Reddit communities (30.1% vs ~16%). Additionally, YouTube’s feedback mechanisms, likes, hearts, and algorithms favoring “agreeable” comments, further suppress negativity and elevate praise.

Together, these platform cultures create distinct emotional ecosystems. Reddit fosters critical evaluation and debate, while YouTube fosters encouragement and affirmational feedback. These sociotechnical differences provide a strong explanatory foundation for the statistically significant sentiment divergences identified in our chi-square tests.

