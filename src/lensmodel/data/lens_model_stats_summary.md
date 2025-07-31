# Lens Model: Statistical Results Summary

## 📊 Model Performance

- **Accuracy**: 59.6%
- **Total Observations**: 156 gender judgments
- **Topics Analyzed**: 20
- **Vocabulary Size**: 356 words (improved from 85)
- **Gender Distribution**: 83 Male, 73 Female judgments

---

## 🔍 Topic Model Results

### **Top 20 Topics with Gender Associations**

| Topic | Coefficient | Odds Ratio | Gender Association | Top Words |
|-------|-------------|------------|-------------------|-----------|
| Topic 16 | +1.490 | 4.438 | Male | mood, music, song, night, listen |
| Topic 3 | +1.267 | 3.552 | Male | assignment, cool, program, haha, season |
| Topic 0 | -0.967 | 0.380 | Female | work, time, lot, probably, game |
| Topic 5 | +0.950 | 2.586 | Male | watch, final, currently, listen, look |
| Topic 18 | -0.811 | 0.444 | Female | general, art, draw, lot, like |
| Topic 17 | -0.811 | 0.444 | Female | day, work, tomorrow, class, short |
| Topic 15 | -0.811 | 0.444 | Female | snack, exam, live, sweet, look |
| Topic 11 | +0.801 | 2.228 | Male | llm, home, summer, want, fun |
| Topic 12 | +0.778 | 2.178 | Male | watch, work, experience, boring, content |
| Topic 14 | -0.615 | 0.540 | Female | production, go, family, home, use |
| Topic 1 | +0.573 | 1.773 | Male | like, song, like like, know, music like |
| Topic 2 | +0.521 | 1.684 | Male | play, game, new, long, fun |
| Topic 4 | +0.489 | 1.631 | Male | question, parent, state world, change, get |
| Topic 6 | +0.456 | 1.578 | Male | club, friend, cafe, honestly, mean |
| Topic 7 | +0.423 | 1.527 | Male | party, little, friend, fun, sure |
| Topic 8 | +0.389 | 1.476 | Male | question, chat, excited, ve, get |
| Topic 9 | +0.356 | 1.428 | Male | bit, know, life, nice, tell |
| Topic 10 | +0.323 | 1.381 | Male | talk, abt, psychology, relationship, run |
| Topic 13 | +0.289 | 1.335 | Male | character, impact, favourite, pretty, book |
| Topic 19 | +0.256 | 1.292 | Male | issue, neuroscience, learn, think, remember |

---

## 🎯 Topic Labels & Interpretations

### **Female-Associated Topics (Negative Coefficients)**

- **Topic 0**: "Work & Gaming" - Work-related discussions and gaming activities
- **Topic 18**: "Art & Creativity" - Artistic activities and creative expression
- **Topic 17**: "Daily Routine" - Day-to-day activities and scheduling
- **Topic 15**: "Food & Exams" - Snacking habits and exam preparation

### **Male-Associated Topics (Positive Coefficients)**

- **Topic 16**: "Music & Mood" - Music preferences and emotional expression
- **Topic 3**: "Academic & Travel" - Academic assignments and travel planning
- **Topic 5**: "Media & Study" - Watching content and studying habits
- **Topic 11**: "Technology & Leisure" - LLM discussions and summer activities
- **Topic 12**: "Content & Experience" - Media consumption and work experience
- **Topic 1**: "Music & Rap" - Music preferences, especially rap
- **Topic 2**: "Gaming & Art" - Video games and artistic activities
- **Topic 4**: "Politics & Family" - Political discussions and family topics
- **Topic 6**: "Social Events" - Club activities and social gatherings
- **Topic 7**: "Parties & Friends" - Party planning and social activities
- **Topic 8**: "Planning & Interests" - Future planning and personal interests
- **Topic 9**: "Life & Communication" - Life discussions and storytelling
- **Topic 10**: "Psychology & Research" - Academic research and psychology
- **Topic 13**: "Literature & Impact" - Books, characters, and personal impact
- **Topic 19**: "Neuroscience & Learning" - Academic study and cognitive topics

---

## 📈 Brunswikian Lens Model Results

### **Cue-Judgment Relationships**

**Strongest Predictors:**
1. **Topic 16** (Music & Mood) → Male judgment (OR: 4.438)
2. **Topic 3** (Academic & Travel) → Male judgment (OR: 3.552)
3. **Topic 0** (Work & Gaming) → Female judgment (OR: 0.380)
4. **Topic 5** (Media & Study) → Male judgment (OR: 2.586)
5. **Topic 18** (Art & Creativity) → Female judgment (OR: 0.444)

### **Classification Performance**
- **Precision**: 59% for both Man and Woman
- **Recall**: 72% for Man, 45% for Woman  
- **F1-Score**: 65% for Man, 51% for Woman

---

## 🔧 Technical Improvements

### **Vocabulary Enhancement**
- **Previous vocabulary**: 85 words (restrictive TF-IDF parameters)
- **Current vocabulary**: 356 words (optimized parameters)
- **Improvement**: 4x increase in vocabulary size
- **Parameters**: `min_df=2`, `max_df=0.95`

### **Topic Quality**
- **More specific topics** with clearer themes
- **Better topic diversity** across conversation domains
- **Maintained model accuracy** despite increased complexity

---

## 🎨 Visualization Files Generated

- `bertopic_umap_clustering.png` - UMAP clustering visualization
- `bertopic_analysis.png` - Topic distribution analysis
- `bertopic_detailed_clustering.png` - Detailed topic clustering
- `bertopic_summary.csv` - Topic summary with top words
- `bertopic_document_topics.csv` - Document-topic probability matrix
- `cue_matrix.csv` - Topic weights merged with gender data
- `lens_model_feature_importance.csv` - Topic importance for gender prediction
- `lens_model_performance.txt` - Model performance summary 