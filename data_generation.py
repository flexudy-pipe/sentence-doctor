from langua import Predict
import pandas as pd
import csv
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import BallTree
import random
import numpy as np

################# SCROLL DOWN TO THE MAIN METHOD ##################


class DataHandler:
    __DEFAULT_LANG = {"de", "en", "fr"}

    __LANGUAGE_DETECTOR = Predict()

    __SENTENCE_MODEL = SentenceTransformer('distiluse-base-multilingual-cased')

    # These transform probabilities greatly influence your dataset. Transforms include
    # Randomly appending words at the beginning and/or end of sentence or randomly breaking a random portion
    # of the beginning and/or end of a sentence
    # Setting the first probability to 1.0 means no sentence will be transformed
    SENTENCE_TRANSFORM_PROBABILITIES = [0.15, 0.5, 0.35]  # don't transform, apply two transform, apply one transform
    # The lengths of SENTENCE_TRANSFORM_PROBABILITIES and NUMBER_OF_TRANSFORMS must match
    NUMBER_OF_TRANSFORMS = [0, 1, 2]

    NUM_WORDS_TO_EXTEND_SENTENCE = [1, 2, 3, 4, 5, 7, 8, 9, 10]

    MAX_SENTENCE_RATIO_TO_BREAK = 0.4

    EXCLUDE_CONTEXT_PROBABILITY = 0.15

    CHARACTER_DISTORTION_PROBABILITY = 0.3

    NUM_CHARS_TO_DISTORT = [1, 2, 3, 4, 5]

    DISTORT_NEIGHBOUR = 0.3

    DESTROY_CHARACTER = 0.5

    def __init__(self, supported_languages: set = None, batch_size: int = 5000):

        self.__supported_languages = supported_languages if supported_languages is not None else self.__DEFAULT_LANG

        self.__batch_size = batch_size

        self.__transforms = [self.__break_sentence_prefix, self.__break_sentence_postfix,
                             self.__extend_sentence_postfix,
                             self.__extend_sentence_prefix]

    def generate_data_set(self, file_path: str, delimiter: str = "\t") -> pd.DataFrame:
        data = self.__load_data(file_path, delimiter)

        data = self.__generate_context(data)

        data = self.__generate_noisy_sentences(data)

        data = self.__generate_final_dataset(data)

        return data

    def __load_data(self, file_path: str, delimiter: str = "\t") -> dict:
        print("Loading the file ...")

        all_data = dict.fromkeys(self.__supported_languages)

        for language in self.__supported_languages:
            all_data[language] = pd.DataFrame(columns=["sentence"]).astype(str)

        data = pd.read_csv(file_path, encoding="utf-8", delimiter=delimiter, quoting=csv.QUOTE_NONE)

        data_frames = self.__filter_unsupported_languages(data)

        for language, data_frame in data_frames.items():
            data_to_stack = pd.DataFrame()

            data_to_stack["sentence"] = data_frame["sentence"]

            all_data[language] = pd.concat([all_data[language], data_to_stack], ignore_index=True).drop_duplicates()

        return all_data

    def __filter_unsupported_languages(self, data: pd.DataFrame) -> dict:
        filtered_sentences = dict.fromkeys(self.__supported_languages)

        for language in self.__supported_languages:
            filtered_sentences[language] = list()

        for _, row in data.iterrows():
            sentence = str(row["sentence"])

            language_is_supported, language = self.__sentence_language_is_supported(sentence)

            if language_is_supported:
                filtered_sentences[language].append({"sentence": sentence})

        data_frames = dict()

        for language, sentences in filtered_sentences.items():
            data_frames[language] = pd.DataFrame(sentences)

        return data_frames

    def __sentence_language_is_supported(self, text: str) -> tuple:
        language = self.__LANGUAGE_DETECTOR.get_lang(text)
        return language in self.__supported_languages, language

    def __generate_context(self, data_frames: dict) -> pd.DataFrame:
        print("generating context ...")

        sentences = list()

        for language, data_frame in data_frames.items():
            sentences.extend(self.__generate_data_frame_context(data_frame))

        return pd.DataFrame(sentences)

    def __generate_data_frame_context(self, data_frame: pd.DataFrame) -> list:
        sentences = list()

        data_frame = data_frame.sample(frac=1.0)

        pointer = 0

        while (pointer + self.__batch_size) <= data_frame.shape[0]:
            batch = data_frame.iloc[pointer:pointer + self.__batch_size]

            sentences.extend(self.__generate_context_from_batch(batch))

            pointer = pointer + self.__batch_size

        sentences.extend(self.__generate_context_from_batch(data_frame.iloc[pointer:]))  # Remaining ( < batch_size )

        return sentences

    def __generate_context_from_batch(self, data_frame_batch: pd.DataFrame) -> list:
        if len(data_frame_batch) < 2:
            return list()  # Just discard the remaining one sentences

        data_rows = list()

        sentences = data_frame_batch.sample(frac=1.0)["sentence"].values

        embeddings = self.__SENTENCE_MODEL.encode(sentences, batch_size=128)

        tree = BallTree(np.array(embeddings), leaf_size=2)

        for i in range(len(embeddings)):
            closest_neighbour_positions = tree.query(embeddings[i:i + 1], k=3, return_distance=False)

            closest_neighbour_positions = closest_neighbour_positions[0]

            closest_neighbour_positions = [neighbour for neighbour in closest_neighbour_positions if neighbour != i]

            random.shuffle(closest_neighbour_positions)

            prefix_context = sentences[closest_neighbour_positions[0]]

            postfix_context = " "

            if len(closest_neighbour_positions) > 1:
                postfix_context = sentences[closest_neighbour_positions[1]]

            data_rows.append({"sentence": sentences[i], "prefix_context": prefix_context,
                              "postfix_context": postfix_context})

        return data_rows

    def __generate_noisy_sentences(self, data: pd.DataFrame) -> pd.DataFrame:
        print("generating noise ...")

        noisy_sentences = list()

        prefix_context = list()

        postfix_context = list()

        i = 0

        for _, row in data.iterrows():
            sentence = str(row["sentence"])

            noisy_sentence, (prefix, postfix) = self.__add_noise_to_sentence(sentence, str(row["prefix_context"]),
                                                                             str(row["postfix_context"]))

            if i % 2 == 0:
                print("[ORIGINAL]" + " " + sentence)

                print("[NOISY]" + " " + noisy_sentence)

                print("original and noisy are the same: " + str(sentence == noisy_sentence))

                print("\n\n")

            noisy_sentences.append(noisy_sentence)

            prefix_context.append(prefix)

            postfix_context.append(postfix)

            i += 1

        data["noisy"] = noisy_sentences

        data["prefix_context"] = prefix_context

        data["postfix_context"] = postfix_context

        return data

    def __add_noise_to_sentence(self, sentence: str, prefix_candidate: str, postfix_candidate: str) -> tuple:
        number_of_transformations = np.random.choice(self.NUMBER_OF_TRANSFORMS, p=self.SENTENCE_TRANSFORM_PROBABILITIES)

        if number_of_transformations == 0:
            return sentence, (prefix_candidate, postfix_candidate)

        transforms = np.random.choice(self.__transforms, number_of_transformations, replace=False,
                                      p=[0.3, 0.3, 0.2, 0.2])

        for transform in transforms:
            if transform == self.__extend_sentence_prefix:
                sentence, prefix_candidate = transform(sentence, prefix_candidate)

            elif transform == self.__extend_sentence_postfix:
                sentence, postfix_candidate = transform(sentence, postfix_candidate)

            elif transform == self.__break_sentence_postfix(sentence):
                sentence, postfix = transform(sentence)

                postfix_candidate = postfix + " " + postfix_candidate

            else:
                sentence, prefix = transform(sentence)

                prefix_candidate = prefix_candidate + " " + prefix

        sentence = self.__distort_characters(sentence)

        prefix_candidate = self.__distort_characters(prefix_candidate)

        postfix_candidate = self.__distort_characters(postfix_candidate)

        exclude_context_probability = random.uniform(0, 1)

        if exclude_context_probability < self.EXCLUDE_CONTEXT_PROBABILITY:
            exclude_context_probability = random.uniform(0, 1)
            if exclude_context_probability < self.EXCLUDE_CONTEXT_PROBABILITY:
                prefix_candidate = ""
                postfix_candidate = ""
            elif random.uniform(0, 1) > 0.5:
                prefix_candidate = ""
            else:
                postfix_candidate = ""

        return sentence, (prefix_candidate, postfix_candidate)

    def __distort_characters(self, text: str) -> str:
        if random.uniform(0, 1) > self.CHARACTER_DISTORTION_PROBABILITY:
            return text

        if len(text) == 0:
            return text

        chars_to_distort = random.choice(self.NUM_CHARS_TO_DISTORT)

        chars = list(text)

        len_text = len(chars)

        for i in range(chars_to_distort):
            position = random.randint(0, len_text-1)

            self.__distort_helper(chars, position, len_text)

            if random.uniform(0, 1) < self.DISTORT_NEIGHBOUR:

                if random.uniform(0, 1) < 0.5:
                    position = position + 1

                else:
                    position = position - 1

                self.__distort_helper(chars, position, len_text)

        return "".join(chars)

    def __distort_helper(self, chars: list, position: int, length: int) -> None:
        if position < 0 or position >= length:
            return

        if random.uniform(0, 1) < self.DESTROY_CHARACTER:
            chars[position] = ""

        else:
            chars[position] = chars[random.randint(0, length-1)]

    def __break_sentence_postfix(self, sentence: str) -> tuple:

        break_percentage = random.uniform(0, self.MAX_SENTENCE_RATIO_TO_BREAK)

        postfix_chars_to_cut = int(len(sentence) * break_percentage)

        postfix = ""

        if 0 < postfix_chars_to_cut < len(sentence):
            postfix = sentence[-postfix_chars_to_cut:]

            sentence = sentence[:-postfix_chars_to_cut]

        return sentence, postfix

    def __break_sentence_prefix(self, sentence: str) -> tuple:
        break_percentage = random.uniform(0, self.MAX_SENTENCE_RATIO_TO_BREAK)

        prefix_chars_to_cut = int(len(sentence) * break_percentage)

        prefix = ""

        if prefix_chars_to_cut > 0:
            prefix = sentence[:prefix_chars_to_cut]

            sentence = sentence[prefix_chars_to_cut:]

        return sentence, prefix

    def __extend_sentence_prefix(self, sentence: str, prefix_candidate: str) -> tuple:

        num_words = np.random.choice(self.NUM_WORDS_TO_EXTEND_SENTENCE)

        prefix_candidate_tokens = prefix_candidate.split()

        truncated_candidate = list()

        if len(prefix_candidate_tokens) > num_words:
            prefix = prefix_candidate_tokens[-num_words:]

            truncated_candidate = prefix_candidate_tokens[:-num_words]

        else:
            prefix = prefix_candidate_tokens

        prefix = " ".join(prefix) + " "

        sentence = prefix + sentence

        return sentence, " ".join(truncated_candidate)

    def __extend_sentence_postfix(self, sentence: str, postfix_candidate: str) -> tuple:
        num_words = np.random.choice(self.NUM_WORDS_TO_EXTEND_SENTENCE)

        postfix_candidate_tokens = postfix_candidate.split()

        truncated_candidate = list()

        if len(postfix_candidate) > num_words:
            postfix = postfix_candidate_tokens[:num_words]

            truncated_candidate = postfix_candidate_tokens[num_words:]

        else:
            postfix = postfix_candidate_tokens

        postfix = " " + " ".join(postfix)

        sentence = sentence + postfix

        return sentence, " ".join(truncated_candidate)

    def __generate_final_dataset(self, data: pd.DataFrame) -> pd.DataFrame:
        data_rows = list()

        lengths = {"source_avg_len": 0, "target_avg_len": 0, "context_avg_len": 0, "noisy_avg_len": 0}

        num_elements = 0

        for _, row in data.iterrows():
            sentence = str(row["sentence"]) + " </s>"

            context = "context: {" + str(row["prefix_context"]) + "}{" + str(row["postfix_context"]) + "} </s>"

            noisy_sentence = "repair_sentence: " + str(row["noisy"]) + " "

            source_text = noisy_sentence + context

            lengths["target_avg_len"] += len(sentence.split())

            lengths["source_avg_len"] += len(source_text.split())

            lengths["context_avg_len"] += len(context.split())

            lengths["noisy_avg_len"] += len(noisy_sentence.split())

            data_rows.append({"source": source_text, "target": sentence})

            num_elements += 1

        for key, value in lengths.items():
            print(key + " " + str(value / num_elements))

            print("\n")

        return pd.DataFrame(data_rows)

    @staticmethod
    def prepare_tatoeba_data(path_to_tatoeba_folder: str, languages: list, max_num_rows_to_process: int = -1) -> str:
        file_names = dict()

        for language in languages:
            # these files have no headers
            file_names[path_to_tatoeba_folder + language + "_sentences.tsv"] = [2]

        return DataHandler.prepare_my_custom_data(file_names, "\t", max_num_rows_to_process=max_num_rows_to_process,
                                                  header=None)

    @staticmethod
    def prepare_my_custom_data(file_paths_and_column_names: dict, delimiter: str, header="infer",
                               max_num_rows_to_process: int = -1) -> str:
        """
        :param header: None if the headers are not in the first line
        :param max_num_rows_to_process: The maximum number of rows you want to read per file.
                                        If your dataset is too large. Value set to -1 will read everything.
        :param file_paths_and_column_names: The collection of file paths and the names of columns your want to
               extract the text from.
               :example > {"file.csv": ["col1", "col4"], "folder/fileX.tsv": ["col2"]}
               We iterate through the dictionary, load the files ("file.csv" and "folder/fileX.tsv") and extract the
               values from the columns ("col1", "col4") for file.csv and ("col2") for folder/fileX.tsv.
               All files must be csv or tsv files
        :param delimiter: can be a comma, tab, semi-colon etc..
        :return: The values of all columns in every file are merged into one column with the name "sentence". The
                 newly created file is saved and its path is returned.
        """

        final_data = pd.DataFrame(columns=["sentence"])

        for path_to_file, columns_with_sentences in file_paths_and_column_names.items():

            if max_num_rows_to_process == -1:
                data = pd.read_csv(path_to_file, encoding="utf-8", delimiter=delimiter, quoting=csv.QUOTE_NONE,
                                   header=header)
            else:
                data = pd.read_csv(path_to_file, encoding="utf-8", delimiter=delimiter, quoting=csv.QUOTE_NONE,
                                   nrows=max_num_rows_to_process, header=header)

            for column_name in columns_with_sentences:
                data_to_stack = pd.DataFrame()

                data_to_stack["sentence"] = data[column_name]

                final_data = pd.concat([final_data, data_to_stack], ignore_index=True)

        new_file_path = "data/merged_sentences.tsv"

        final_data.to_csv(new_file_path, encoding="utf-8", sep="\t", index=False)

        return new_file_path


if __name__ == "__main__":
    
    # ALL YOU NEED TO UNDERSTAND IS HERE BELOW. BUT YOU ARE FREE TO READ THE CODE AND MAKE CHANGES
    
    # The data handler will take care of creating your training data and make it ready for the train_any_t5_task.py found in the project
    data_handler = DataHandler()

    # However, the Data Handler needs your data to be in a particular format. An easy one actually ;)
    # A tsv or csv file with one column containing all the sentences. (Language don't matter). Just mix everything under one column
    # The header of the column is "sentence"
    
    # In this example, i downloaded the french sentences from tatoeba. So I want to prepare my dataset
    # before sending it to the data handler. 
    
    # TODO provide the folder name containing your data and the tatoeba language keys for each file you downloaded
    #      Since i only have french in this example, I pass in ["fra"] as languages. If i had english and german too, I 
    #      would have passed `languages=["deu", "eng", "fra"]`
    
    file_path = DataHandler.prepare_tatoeba_data("data/", languages=["fra"],
                                                 max_num_rows_to_process=100000)
                                                 
    # Ofcourse you don't need any data preparation if you already have a csv/tsv file containing one column with all your sentences :)
    
    # The data preparation is however practical if you have multiple files containing sentences from multiple columns you want to use for training
    # Since I am lazy, i wrote a generic function to do all the preprocessing. For example:
    # Say you file A.tsv with columns c1 and c2 containing sentences you want to use for training. You also have a file B.tsv with
    # columns c3 and c4. Then you can prepare the data in a single step like this:
    
    # file_path = DataHandler.prepare_my_custom_data({"data/A.tsv": ["c1", "c2"], "data/B.tsv": ["c3", "c4"]})
    
    # This will produce a file "data/merged_sentences.tsv". The path is returned to the variable file_path.
    
    # Well it is done. Just pass the file path to the data handler generate function like this
    data_set = data_handler.generate_data_set(file_path)

    print(data_set.head(n=10))

    # Save the generated file wherever you want
    data_set.to_csv("data/sentence_doctor_dataset_300.csv", encoding="utf-8", sep="\t", index=False)
    
    # I assume you have a GPU cuz if not, ... better grab cups of coffees for the next hours or days. :D 
