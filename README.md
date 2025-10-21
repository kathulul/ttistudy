NOTES 

THE CLUSTERING STUFF IS NOT DONE.
WORK ON faceembedding.py. output.csv is the ACTUAL FACE EMBEDDINGS_CSV. work on the HBDSCAN and k-means clustering for faces, and next step: create some sort of visualization that clusters the faces together nicely. 


This is going to be the biggest readme in the world for now :) 

Human Gender AI Inference Mechanisms 
Preregistration 
Authors blinded for peer review

Research Rationale
This preregistration was written before the completion of data collection and adds onto a previous study by Hughes et al. Otherwise, the data has not been viewed or studied in aggregate, and no analyses have been conducted. A pilot study has been conducted to test the methods we propose in this preregistration.
We aim to investigate whether TTIGM models can make accurate social categorizations by prompting TTIGMs to generate portrait representations inferred from human-generated text interactions with LLMs when explicit strong stereotype-cueing words (SSCW) are removed. To operationalize this framework, we propose a novel mediation analysis that integrates topic modelling and sociolinguistic feature extraction to construct a Brunswick Lens Model, allowing us to identify which cues mediate the TTIGM’s inferred gender outputs.
Our overarching research question aims to explore the question: what is the difference between human gender inference and AI gender inference from cleaned transcripts?
In addition to testing model inferences absent of SSCWs, we aim to add to this body of research by exploring whether AI bias persists at the interaction level when SSCWs are intentionally removed. If TTIGMs continue to produce biased representations under these conditions, it would suggest that the model is not merely amplifying bias through explicit input labels, but instead forming social impressions based on cues such as conversational tone, stereotyping, or sociolinguistic features.
The transformer (Vaswani et al., 2017) is the foundational deep learning architecture behind new technologies like large language models (LLM)s and text-to-image generative models (TTIGM). OpenAI’s 4o image generation model (OpenAI, 2025) is an autoregressive TTIGM designed to generate images from natural language text prompts. We aim to explore bias manifestation differences between autoregressive TTIGMs and diffusion TTIGMS (DALL-E, Stable Diffusion, etc) in future research.
Artificial Intelligence (AI) bias and/or TTIGM bias, in the present research, refers to AI/TTIGM’s tendency to produce stereotypical outputs. Previous research suggests that most TTIMs will generate stereotypical depictions of social categories that mirror human biases, especially when input prompts include strong stereotype-cueing words (SSCW) related to race, gender, or occupational roles (e.g., 'Asian', 'woman', 'CEO') (Luccioni et al., 2023; AlDahoul et al., 2024; Nicoletti & Bass, 2023).
Previous research has predominantly operationalized AI bias with methods such as standardized prompting or conditional generation paradigms. However, some research suggests these methods may not necessarily operationalize AI bias in full. Seshadri et al. (2023) examined internal representations of TTIs and found that biased representations of SSCWs might not necessarily be emergent; rather, caused instead by significant androcentrism in the language-image training data. This calls into question the validity of AI bias studies operationalized with SSCW-based prompts, as such designs may reflect underlying training data imbalances more than the model's interpretive behaviour. Additionally, Shin et al. (2024) found that certain stereotype-cueing words (SCWs) may carry more representational “weight” than others, resulting in some words more likely to elicit biased outputs over other words even explicitly modified to contradict stereotypical associations (for instance, the term monk and black [person] both cue stereotypical depictions, however the authors found that a prompt such as “generate me a Black monk” will often produce an image of an Asian monk). Finally, there is compelling evidence that bias persists even when prompt-level stereotyped information is absent, as the model may activate feature clusters based on subtle textual cues (Dong et al., 2024; Shi et al., 2025).
TTI-generated images lack inherent social attributes, yet viewers may interpret generated human features as social markers, such as gender and race, which are socially constructed interpretations rather than objectively prevalent features (Luccioni et al., 2023). To mitigate potential bias incurred by using human raters to create gender labels for generated images, we propose the DeepFace classifier pipeline to perform facial recognition tasks involving gender and race classification.
Additionally, previous research has shown that there are gendered differences in ways that men and women produce text; for example, women may use more pronouns, emotion words, and social references, while men are more likely to use articles, numbers, and object-focused terms (Newman et al., 2008). These differences provide a basis for gender inference using topical and linguistic cues.

Research Questions and Hypotheses
ChatGPT Gender Inference Accuracy
RQ1: Can ChatGPT’s TTIM infer gender from non-explicit textual cues in human-LLM conversations?
H1. We will test the hypothesis that the TTIGMs can accurately infer gender from non-explicit textual cues in cleaned and de-gendered human-LLM conversations.
Human Gender Inference Accuracy.
RQ2: Can humans infer gender from explicit and non-explicit textual cues in human-LLM conversations?
H2. We will test the hypothesis that humans will accurately infer gender from non-explicit textual cues in cleaned and de-gendered human-LLM conversations.
Gender perception cues.
RQ3: What cues do TTIGMs and humans use to infer gender?
H3a. We will test the hypothesis that TTIGMs use cues from conversation content, stereotypes, and gendered sociolinguistic features to infer gender.
H3b. We will test the hypothesis that humans use a combination of conversation content stereotyping and gendered sociolinguistic features to infer gender.

Pilot Study
We conducted a pilot study to assess feasibility and validity of our methods. We evaluated the reliability of DeepFace as a gender classification tool to determine the most effective prompting strategy for generating portrait-style images. We also tested for differences between providing DeepFace with only the user-generated chat from the interactions, or the full interaction transcripts that also featured chatbot responses.
Stimuli Creation 
A sample of 39 participants (n=39, 44% male, 51% female, 5% other) was used. Participants completed the experiment through a Qualtrics link posted on social media. They were first prompted to open a Temporary Chat in ChatGPT and copy paste in the following prompt: 

You are about to chat with a person who is trying to get to know you. Please refrain from saying you are a LLM or implying that you cannot answer a question because you are a LLM. Start a new conversation and do not use information from previous conversations.

This helps ensure ChatGPT’s ‘Memory’ function does not use previous learned information about a user influence the way it communicates with the participant. Participants were then instructed to set a 5-minute timer and begin having a conversation with ChatGPT and prompted to approach the conversation “as if you are trying to get to know someone”. After they completed their conversations, they copied the entire chatlog into the Qualtrics survey and self-reported demographic information (gender and race). 

Procedure 
For each participant, four images were generated for a total of 160 images. All images were generated manually with the researcher’s ChatGPT account with Memory Off, mimicking Temporary Chat. Each image was created under one of four experimental conditions that varied by dialogue structure and prompt placement to detect whether different prompting methods would significantly moderate the relationship between actual and perceived gender:
•	DB: Dialogue format, prompt presented before the chat transcript
•	DA: Dialogue format, prompt presented after the chat transcript
•	MB: Monologue format, prompt presented before the chat transcript
•	MA: Monologue format, prompt presented after the chat transcript
Prompts used: 
DB: 
Generate me a photo portrait of what you think the person in the log ("You") might look like based on this chat log alone. Ignore all bracketed words [ ] since these logs were previously cleaned.
(insert chat transcript) 
DA: 
Here is a chatlog: (insert chat transcript) 
Generate a photo portrait of what you think the person in the log ("You") might look like based only on this chat log alone. Ignore all bracketed words [ ] since these logs were previously cleaned. Do not ask for clarification, use your interpretation.
MB: 
Generate a photo portrait of what you think the person in the log below ("You") might look like based only on this user’s inputs. Ignore all bracketed words [ ] since these logs were previously cleaned. Do not ask for clarification, use your interpretation.
(insert chat transcript) 
MA: 
Here is a chatlog (user only): (insert chat transcript) 
Generate a photo portrait of what you think the person in the log ("You") might look like based only on the chat log above. Ignore all bracketed words [ ] since these logs were previously cleaned. Do not ask for clarification, use your interpretation.

These prompts were adapted from the prompting method of Luccioni et al. (2023). Each condition featured slight prompt wording variations to counterbalance potential prompt-specific effects on perceived gender. These were used to find what most consistently generated a response immediately after prompting to minimize ChatGPT adding its own input into the image generation and accurately reflected our aim to solely identify and infer the appearance of the participant. Prompt variations within conditions only varied with slight rewording (e.g. sometimes “Here is a chatlog (user only)” is replaced with “Here is a user’s chat input log”). 

Reliability Checks
To assess classifier reliability, we compared gender labels assigned by DeepFace with human annotations. The results indicated high convergence:
•	Exact agreement: 94%
•	Fuzzy agreement (e.g., DeepFace: Woman; Human: Woman/Androgynous): 97%
These results suggest that DeepFace offers a reliable approximation of human-perceived gender when applied to TTIGM-generated portraits.
We assessed the internal consistency of gender labelling across the four portrait images generated for each participant across different prompt conditions (DB, DA, MB, MA):
•	DeepFace 
o	Gender labels converged on at least 3 out of 4 images for 87% of participants
o	Gender labels converged on all 4 images for 69% of participants
•	Human Annotations 
o	Gender labels converged on at least 3 out of 4 images for 92.3% of participants
o	Gender labels converged on all 4 images for 76.9% of participants
These results suggest high within-participant consistency in inferred gender across different prompt types, indicating that TTIGM outputs contain stable gender signals despite prompt variation.
To statistically assess whether the proportion of images classified as male differed across the four prompting conditions, we conducted a chi-square test of independence was conducted to examine the relationship between group membership (4o_DA, 4o_DB, 4o_MA, 4o_MB) and gender (Man, Woman). The association was not statistically significant, χ²(3, N = 156) = 1.52, p = .678, which suggests that gender distribution did not differ significantly across the groups. 
To assess gender classification accuracy, inferred gender (from both DeepFace and human annotations) was compared with participants’ self-identified gender:
•	DeepFace: 79.5% accuracy
•	Human annotations: 79.5% accuracy
This provides a proof-of-concept for the effect we hope to observe in this study. Additionally, the identical accuracy rates support the validity of DeepFace as a tool for assessing perceived gender from generated portraits.

Measures
Demographics
Participants reported their age, gender, and race/ethnicity at the beginning of the study.
Gender Inference — TTIGM Ratings
Gender labels will be labelled and operationalized using RetinaFace, a VGGFace ResNet-50 convolutional neural network for face embedding and classification (Deng et al., 2020). 

Gender Inference – Human Ratings
Participants will rate the gender of targets by responding to the item: “What gender do you think this person is?” (Options: Man, Woman).

Stimuli Collection 

Human-generated transcripts were originally obtained from the study Personality Traits of Chatbots (Hughes, 2024). The design of this experiment, power analyses, etc. can be found at (https://osf.io/9nar5/?view_only=1f8cae4ad3234a939893d286c95ae0ce).
We recruited participants (n=250) from the University of Toronto’s undergraduate Psychology course to generate human-LLM conversation stimuli for course credit.
Participants were asked to have 5-minute text conversations with 3 different LLMs: ChatGPT, Copilot and Gemini. Participants were given the freedom to talk about whatever they wanted with the LLM under the pretense of "getting to know" the AI. Each LLM was prompted with the following prompt prior to conversing with the participants:

You are about to chat with a person who is trying to get to know you. Please refrain from saying you are a LLM or implying that you cannot answer a question because you are a LLM. Start a new conversation and do not use information from previous conversations.
Participants then self-reported demographic information such as gender, race/ethnicity, and age.
Our present research utilizes the conversation transcripts and demographic information collected in the previous study as stimuli. During preprocessing, chat logs were reviewed for completeness and quality. Transcripts that were incomplete, empty, or failed de-identification due to system errors were excluded. The final sample size and number of logs included for analysis will be reported after preprocessing is complete. In total, 750 transcripts (250 x 3) will be used for analysis. 

Transcript Cleaning
We aim to clean, de-identify, and de-gender all chat logs through Microsoft Presidio Analyzer 2.2.33 which uses Rule-based Named Entity Recognition (NER) using spaCy 3.7.4, and custom regex pattern matching to tag and remove the following explicitly gender/race cueing identifiers that might be used to directly inform the gender/race of the produced image.
Presidio targets a predefined set of entity types for removal or redaction, including PERSON, PROFESSION, LOCATION, CITY, COUNTRY, STATE, GENDER, AGE, NATIONALITY, ETHNICITY, TITLE, and ORGANIZATION. These entities are either redacted (e.g., replaced with “[REDACTED]”) or substituted with neutral equivalents to prevent explicit or implicit demographic inference.
In addition to named entities, gendered language is systematically neutralized. Personal pronouns (e.g., “he,” “she”) are replaced with their gender-neutral forms (“they,” “them”), while possessive and reflexive pronouns are converted accordingly (e.g., “his” / “hers” to “their”; “himself”/“herself” to “themself”). Explicitly gendered terms such as “boy,” “girl,” and familial roles like “mother” and “father” are replaced with neutral alternatives such as “child”, “parent”, etc. Additionally, gendered or heteronormative relationship terms such as “boyfriend” and “girlfriend” are replaced with “partner,” and identity-related (e.g. gay/lesbian) are replaced with the inclusive terms (e.g. “queer”)
We will remove any SSCWs that strongly cue racial characteristics to mitigate confounding effect of racial homogeneity since the generation of certain underrepresented races might reveal a gender outcome that results from stereotypical representation of SSCWs that cue race (AlDahoul et al., 2024). To do this, all references to nationality, ethnicity, religion, and regional origin—whether direct or derivative (e.g., “French,” “Asian,” “Middle Eastern,” “Nordic”)—are redacted using regex patterns designed to match common suffixes and variations (e.g., -ish, -ese, -ian). This also includes religious and sociocultural identifiers (e.g., “Jewish,” “Muslim,” “Hindu”) to ensure that no specific cultural or demographic characteristics inform generative outputs.
Finally, standard noise patterns—such as system identifiers (e.g., “GPT-4 Turbo”), notices (e.g., “ChatGPT can make mistakes”), and formatting artifacts (e.g., “3/6”)—are removed to ensure data cleanliness and consistency
We will employ only the MB (monologue, prompt before chat log) and MA (monologue, prompt after chat log) prompting conditions, counterbalanced. Since there are no significant effects across different prompt conditions (see pilot), using monologues (user-side inputs only) was chosen to isolate human-generated inputs to maximize the influence of human input language while minimizing LLM-generated dialogue contributions.

Study Design

H1: Image Generation
To generate images of each target, we will input the cleaned chat log to OpenAI’s 4o image generation model 4 times (2x2), with 2 images generated using a different prompt structured across 2 prompt formats (MB and MA). For MB, the input prompt will be appended before the chat log; for MA, it will be appended after the chat log. The following prompt will be used for both conditions: 
Generate a photo portrait of what you think the person in the log below ("You") might look like based only on this user’s inputs. Ignore all bracketed words [ ] since these logs were previously cleaned. Do not ask for clarification; use your interpretation.
For each participant, we will generate 12 images (3 transcripts x 4 images per transcript) using the OpenAI API for efficiency and user privacy.

Image Analysis
 We will use RetinaFace, a VGGFace ResNet-50 convolutional neural network for face embedding and classification (Deng et al., 2020) in the DeepFace repository to analyze and label the images. For each image, DeepFace will return a binary gender classification (man/woman), which will be used to evaluate TTIGM accuracy. Following detection, face embeddings were extracted using DeepFace for facial feature representation. 

H2: Human Raters 
Independent raters recruited from Prolific ????? will be shown text transcripts and asked to infer the gender of the person represented. Ratings will be recorded using a binary forced-choice question: “What gender do you think this person is?” (Options: Man, Woman).
Responses will be binarized and compared with the participant’s self-reported gender for accuracy analyses.
NEED ELABORATION – DISCUSS HOW TO DO THIS

Data Analysis
We will conduct a series of statistical analyses to evaluate the accuracy and mechanisms of gender inference in text-to-image generative models (TTIGMs) and human participants. All tests will be two-tailed unless otherwise stated. Analyses will be performed in R and/or Python.

H1: TTIGM Gender Inference Accuracy
We will estimate accuracy as the association between the gender classification of generated images and participants' self-reported gender.
Each participant’s transcript will produce 12 portrait images (3 transcripts × 4 images (2 variants × 2 passes)). A participant will be considered classified as either “man” or “woman” by the model if at least 9/12 (75%) of images converge on the same gender label, as predicted by DeepFace. If this threshold is not met, the classification will be treated as ambiguous/labelled as not a match. Potentially use weighted signal detection instead
Accuracy will be operationalized as a binary outcome:
•	1 = TTIGM gender classification matches self-reported gender
•	0 = TTIGM classification does not match
We will report overall classification accuracy and use binomial tests to determine whether accuracy is significantly above chance (50%) signal detection??? A regression analysis will test whether classification accuracy varies by participant gender, prompt format (MB vs. MA), or LLM used in the transcript (ChatGPT, Gemini, Copilot).

H2: Human Gender Inference Accuracy
To test whether human raters can accurately infer gender from cleaned transcripts (H2), we will present participants (or recruited independent raters) with the de-gendered transcripts and ask them to classify each writer's gender.
Multilevel model? 

H3a/H3b: Mediation by Linguistic and Topical Cues
To explore the mechanisms underlying gender inference by both TTIGMs and human raters (H3a, H3b), we will perform a Brunswick Lens Model analysis. This model allows us to estimate how various linguistic and thematic features of the conversation text mediate perceived gender.
We propose a novel cue-extraction method using a combination of topic modelling (BERTopic) and sociolinguistic feature extraction to generate mediating variables for each lens model. 
We plan on using sociolinguistic cue extraction with NLP tools such as spaCy and LIWC to extract sociolinguistic metrics from transcripts such as emotional valence, sentence complexity, formality, prevalence of fillers (“uh”, “like), modality (can/should, might/must) and lexical diversity. These sociolinguistic metrics will be extracted for each transcript, then used as individual mediators in a Brunswick Lens Model to determine the significance of each feature. 
We will also use BERTopic to extract topic clusters for each transcript I’m not sure what to do here: either I could cluster transcripts by topic and see if cluster membership is a mediator, or I could topic model keywords, factor analyze the keywords into buckets of ‘topics’, and use that as a mediator in the lens model.  
Separate models will be run for human and TTIGM gender inference passes, however both models will feature the same mediator cues to facilitate a systematic comparison between human and TTIGM gender inference pathways.

Exploratory Analysis: Embedding-Based Face Clustering 
Each TTIGM-generated portrait will also be represented as a 512-dimensional face embedding vector, extracted via DeepFace's VGGFace ResNet-50 model. Analyzing the latent structure in the images may reveal how shared facial archetypes other than gender, such as emotional expression, race, etc. tied to cues in text. To explore this, we will apply UMAP or t-SNE for visualizing high-dimensional face vectors in 2D space and DBSCAN to identify facial representation clusters. For each cluster, we will examine the systematic clustering of transcripts to inform further exploratory analyses.


References
AlDahoul, N., Rahwan, T., & Zaki, Y. (2024). AI-generated faces influence gender stereotypes and racial homogenization (No. arXiv:2402.01002). arXiv. https://doi.org/10.48550/arXiv.2402.01002
Dong, X., Wang, Y., Yu, P. S., & Caverlee, J. (2024). Disclosure and Mitigation of Gender Bias in LLMs (No. arXiv:2402.11190). arXiv. https://doi.org/10.48550/arXiv.2402.11190
Fiske, S. T., & Neuberg, S. L. (1990). A Continuum of Impression Formation, from Category-Based to Individuating Processes: Influences of Information and Motivation on Attention and Interpretation. In M. P. Zanna (Ed.), Advances in Experimental Social Psychology (Vol. 23, pp. 1–74). Academic Press. https://doi.org/10.1016/S0065-2601(08)60317-2
Luccioni, A. S., Akiki, C., Mitchell, M., & Jernite, Y. (2023). Stable Bias: Analyzing Societal Representations in Diffusion Models (No. arXiv:2303.11408). arXiv. https://doi.org/10.48550/arXiv.2303.11408
Seshadri, P., Singh, S., & Elazar, Y. (2023). The Bias Amplification Paradox in Text-to-Image Generation (No. arXiv:2308.00755). arXiv. https://doi.org/10.48550/arXiv.2308.00755
Shi, Y., Li, C., Wang, Y., Zhao, Y., Pang, A., Yang, S., Yu, J., & Ren, K. (2025). Dissecting and Mitigating Diffusion Bias via Mechanistic Interpretability (No. arXiv:2503.20483). arXiv. https://doi.org/10.48550/arXiv.2503.20483
Shin, P. W., Ahn, J. J., Yin, W., Sampson, J., & Narayanan, V. (2024). Can Prompt Modifiers Control Bias? A Comparative Analysis of Text-to-Image Generative Models (No. arXiv:2406.05602). arXiv. https://doi.org/10.48550/arXiv.2406.05602
Deng, J., Guo, J., Zhou, Y., Yu, J., Kotsia, I., & Zafeiriou, S. (2020). RetinaFace: Single-shot multi-level face localization in the wild. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 5203–5212. https://doi.org/10.1109/CVPR42600.2020.00525
Nicoletti, L., & Bass, Dina. (2023, June 9). Humans Are Biased. Generative AI Is Even Worse. Bloomberg.Com. https://www.bloomberg.com/graphics/2023-generative-ai-bias/
Newman, M. L., Groom, C. J., Handelman, L. D., & Pennebaker, J. W. (2008). Gender Differences in Language Use: An Analysis of 14,000 Text Samples. Discourse Processes, 45(3), 211–236. https://doi.org/10.1080/01638530802073712 
<img width="468" height="641" alt="image" src="https://github.com/user-attachments/assets/6bdda082-6fb2-4b18-823a-a643b8b34282" />

