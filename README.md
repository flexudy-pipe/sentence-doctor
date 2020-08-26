![avatar](sent-banner.png)

# Sentence-Doctor
Sentence doctor is a T5 model that attempts to correct the errors or mistakes found in sentences. Model works on English, German and French text.

## 1. Problem:
Many NLP models depend on tasks like *Text Extraction Libraries, OCR, Speech to Text libraries* and **Sentence Boundary Detection**
As a consequence errors caused by these tasks in your NLP pipeline can affect the quality of models in applications. Especially since models are often trained on **clean** input.

## 2. Solution:
Here we provide a model that **attempts** to reconstruct sentences based on the its context (sourrounding text). The task is pretty straightforward:
* `Given an "erroneous" sentence, and its context, reconstruct the "intended" sentence`.

## 3. Use Cases:
* Attempt to repair noisy sentences that where extracted with OCR software or text extractors.
* Attempt to repair sentence boundaries.
  * Example (in German): **Input: "und ich bin im**", 
    * Prefix_Context: "Hallo! Mein Name ist John", Postfix_Context: "Januar 1990 geboren."
    * Output: "John und ich bin im Jahr 1990 geboren"
* Possibly sentence level spelling correction -- Although this is not the intended use.
 * Input: "I went to church **las yesteday**" => Output: "I went to church last Sunday".
 
## 4. Disclaimer
Note how we always emphises on the word *attempt*. The current version of the model was only trained on **150K** sentences from the tatoeba dataset: https://tatoeba.org/eng. (50K per language -- En, Fr, De).
Hence, we strongly encourage you to finetune the model on your dataset. We might release a version trained on more data.


