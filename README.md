![avatar](sent-banner.png)

# sentence-doctor
Sentence doctor is a T5 model that attempts to correct the errors or mistakes found in sentences. Model works on English, German and French text.

## Problem:
Many NLP models depend on tasks like *Text Extraction Libraries, OCR, Speech to Text libraries* and **Sentence Boundary Detection**
As a consequence errors caused by these tasks in your NLP pipeline can affect the quality of models in applications. Especially since models are often trained on **clean** input.

## Solution:
Here we provide a model that **attempts** to reconstruct sentences based on the its context (sourrounding text). The task is pretty straightforward:
* `Given an "erroneous" sentence, and its context, reconstruct the "intended" sentence`.






