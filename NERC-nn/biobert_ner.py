import json
import math
import os

import numpy as np
import tensorflow as tf

from dataset import Dataset


DEFAULT_MODEL_NAME = "dmis-lab/biobert-base-cased-v1.1"
IGNORE_INDEX = -100


def _require_transformers():
    try:
        from transformers import AutoTokenizer, TFAutoModelForTokenClassification, create_optimizer
    except ImportError as exc:
        raise ImportError(
            "BioBERT requires the `transformers` package. Install it with "
            "`pip install transformers` before importing this module in Colab."
        ) from exc

    return AutoTokenizer, TFAutoModelForTokenClassification, create_optimizer


def _ordered_labels(labels):
    remaining = set(labels)
    ordered = []

    if "O" in remaining:
        ordered.append("O")
        remaining.remove("O")

    for prefix in ("B-", "I-"):
        current = sorted(label for label in remaining if label.startswith(prefix))
        ordered.extend(current)
        remaining.difference_update(current)

    ordered.extend(sorted(remaining))
    return ordered


def dataset_to_examples(data):
    examples = []
    for sid in data.sentence_ids():
        sentence = data.get_sentence(sid)
        examples.append(
            {
                "id": sid,
                "tokens": [token["form"] for token in sentence],
                "labels": [token["tag"] for token in sentence],
            }
        )

    return examples


@tf.keras.utils.register_keras_serializable(package="biobert_ner")
def masked_sparse_categorical_crossentropy(y_true, y_pred):
    y_true = tf.cast(y_true, tf.int32)
    valid_positions = tf.not_equal(y_true, IGNORE_INDEX)
    safe_y_true = tf.where(valid_positions, y_true, tf.zeros_like(y_true))

    losses = tf.keras.losses.sparse_categorical_crossentropy(
        safe_y_true,
        y_pred,
        from_logits=True,
    )
    valid_positions = tf.cast(valid_positions, losses.dtype)

    total_loss = tf.reduce_sum(losses * valid_positions)
    total_items = tf.reduce_sum(valid_positions)

    return tf.math.divide_no_nan(total_loss, total_items)


@tf.keras.utils.register_keras_serializable(package="biobert_ner")
def masked_sparse_categorical_accuracy(y_true, y_pred):
    y_true = tf.cast(y_true, tf.int32)
    valid_positions = tf.not_equal(y_true, IGNORE_INDEX)
    safe_y_true = tf.where(valid_positions, y_true, tf.zeros_like(y_true))

    predicted = tf.argmax(y_pred, axis=-1, output_type=tf.int32)
    matches = tf.cast(tf.equal(predicted, safe_y_true), tf.float32)
    valid_positions = tf.cast(valid_positions, tf.float32)

    return tf.math.divide_no_nan(
        tf.reduce_sum(matches * valid_positions),
        tf.reduce_sum(valid_positions),
    )


class BioBERTNER:
    def __init__(
        self,
        model_name=DEFAULT_MODEL_NAME,
        max_length=512,
        label_all_tokens=False,
        from_pt=True,
    ):
        self.model_name = model_name
        self.max_length = max_length
        self.label_all_tokens = label_all_tokens
        self.from_pt = from_pt

        self.tokenizer = None
        self.model = None
        self.label_list = None
        self.label2id = None
        self.id2label = None

    def _ensure_tokenizer(self):
        if self.tokenizer is None:
            AutoTokenizer, _, _ = _require_transformers()
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)

    def _build_label_space(self, examples):
        labels = []
        for example in examples:
            labels.extend(example["labels"])

        self.label_list = _ordered_labels(labels)
        self.label2id = {label: idx for idx, label in enumerate(self.label_list)}
        self.id2label = {idx: label for label, idx in self.label2id.items()}

    def _build_model(self):
        self._ensure_tokenizer()
        _, TFAutoModelForTokenClassification, _ = _require_transformers()

        try:
            self.model = TFAutoModelForTokenClassification.from_pretrained(
                self.model_name,
                from_pt=self.from_pt,
                num_labels=len(self.label_list),
                id2label=self.id2label,
                label2id=self.label2id,
                ignore_mismatched_sizes=True,
            )
        except Exception as exc:
            raise RuntimeError(
                "Unable to load the BioBERT checkpoint. If you are using Colab, make sure "
                "internet access is enabled and that PyTorch is available because this "
                "checkpoint is typically loaded with `from_pt=True`."
            ) from exc

    def _align_labels(self, word_labels, word_ids):
        label_ids = []
        previous_word_id = None

        for word_id in word_ids:
            if word_id is None:
                label_ids.append(IGNORE_INDEX)
            elif word_id != previous_word_id:
                label_ids.append(self.label2id[word_labels[word_id]])
            elif self.label_all_tokens:
                label_ids.append(self.label2id[word_labels[word_id]])
            else:
                label_ids.append(IGNORE_INDEX)

            previous_word_id = word_id

        return label_ids

    def _encode_examples(self, examples, include_labels):
        self._ensure_tokenizer()

        encoded_inputs = {
            "input_ids": [],
            "attention_mask": [],
        }
        encoded_labels = []
        encoded_word_ids = []
        has_token_type_ids = None

        for example in examples:
            tokenized = self.tokenizer(
                example["tokens"],
                is_split_into_words=True,
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_attention_mask=True,
            )
            word_ids = tokenized.word_ids()

            encoded_inputs["input_ids"].append(tokenized["input_ids"])
            encoded_inputs["attention_mask"].append(tokenized["attention_mask"])
            encoded_word_ids.append(word_ids)

            if has_token_type_ids is None:
                has_token_type_ids = "token_type_ids" in tokenized
                if has_token_type_ids:
                    encoded_inputs["token_type_ids"] = []

            if has_token_type_ids:
                encoded_inputs["token_type_ids"].append(tokenized["token_type_ids"])

            if include_labels:
                encoded_labels.append(self._align_labels(example["labels"], word_ids))

        numpy_inputs = {
            key: np.asarray(value, dtype=np.int32) for key, value in encoded_inputs.items()
        }

        if include_labels:
            return numpy_inputs, np.asarray(encoded_labels, dtype=np.int32), encoded_word_ids

        return numpy_inputs, encoded_word_ids

    def _build_tf_dataset(self, inputs, labels=None, batch_size=8, shuffle=False):
        if labels is None:
            dataset = tf.data.Dataset.from_tensor_slices(inputs)
        else:
            dataset = tf.data.Dataset.from_tensor_slices((inputs, labels))

        if shuffle:
            first_key = next(iter(inputs))
            dataset = dataset.shuffle(
                buffer_size=max(1, len(inputs[first_key])),
                reshuffle_each_iteration=True,
            )

        return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    def compile(self, train_size, batch_size=8, epochs=3, learning_rate=2e-5, weight_decay=0.01):
        if self.model is None:
            raise ValueError("The model must be created before calling compile().")

        _, _, create_optimizer = _require_transformers()

        steps_per_epoch = max(1, math.ceil(train_size / batch_size))
        optimizer, _ = create_optimizer(
            init_lr=learning_rate,
            num_train_steps=max(1, steps_per_epoch * epochs),
            weight_decay_rate=weight_decay,
            num_warmup_steps=0,
        )

        self.model.compile(
            optimizer=optimizer,
            loss=masked_sparse_categorical_crossentropy,
            metrics=[masked_sparse_categorical_accuracy],
        )

    def fit(
        self,
        train_dir,
        validation_dir=None,
        epochs=3,
        batch_size=8,
        learning_rate=2e-5,
        weight_decay=0.01,
        callbacks=None,
    ):
        train_data = Dataset(train_dir)
        train_examples = dataset_to_examples(train_data)

        if not train_examples:
            raise ValueError(f"No training sentences were found in {train_dir!r}.")

        self._build_label_space(train_examples)
        self._build_model()

        train_inputs, train_labels, _ = self._encode_examples(train_examples, include_labels=True)
        train_dataset = self._build_tf_dataset(
            train_inputs,
            train_labels,
            batch_size=batch_size,
            shuffle=True,
        )

        validation_dataset = None
        if validation_dir is not None:
            validation_data = Dataset(validation_dir)
            validation_examples = dataset_to_examples(validation_data)
            validation_inputs, validation_labels, _ = self._encode_examples(
                validation_examples,
                include_labels=True,
            )
            validation_dataset = self._build_tf_dataset(
                validation_inputs,
                validation_labels,
                batch_size=batch_size,
                shuffle=False,
            )

        self.compile(
            train_size=len(train_examples),
            batch_size=batch_size,
            epochs=epochs,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
        )

        history = self.model.fit(
            train_dataset,
            validation_data=validation_dataset,
            epochs=epochs,
            callbacks=callbacks or [],
            verbose=1,
        )

        return history

    def predict_examples(self, examples, batch_size=8):
        if self.model is None or self.tokenizer is None:
            raise ValueError("Load or train a model before calling predict_examples().")

        inputs, word_ids_by_example = self._encode_examples(examples, include_labels=False)
        predict_dataset = self._build_tf_dataset(inputs, labels=None, batch_size=batch_size, shuffle=False)

        token_predictions = []
        for batch_inputs in predict_dataset:
            logits = self.model(batch_inputs, training=False).logits
            token_predictions.extend(
                tf.argmax(logits, axis=-1, output_type=tf.int32).numpy().tolist()
            )

        predictions = []
        for example, token_ids, word_ids in zip(examples, token_predictions, word_ids_by_example):
            word_labels = ["O"] * len(example["tokens"])
            assigned_words = set()

            for token_label_id, word_id in zip(token_ids, word_ids):
                if word_id is None or word_id in assigned_words or word_id >= len(word_labels):
                    continue

                word_labels[word_id] = self.id2label[int(token_label_id)]
                assigned_words.add(word_id)

            predictions.append(word_labels)

        return predictions

    def predict_dataset(self, datadir, batch_size=8):
        data = Dataset(datadir)
        examples = dataset_to_examples(data)
        predictions = self.predict_examples(examples, batch_size=batch_size)
        return data, predictions

    def predict_to_file(self, datadir, outfile, batch_size=8):
        data, predictions = self.predict_dataset(datadir, batch_size=batch_size)
        output_entities(data, predictions, outfile)
        return predictions

    def evaluate(self, datadir, outfile, batch_size=8):
        predictions = self.predict_to_file(datadir, outfile, batch_size=batch_size)
        evaluate_predictions(datadir, outfile)
        return predictions

    def save(self, save_directory):
        if self.model is None or self.tokenizer is None or self.label_list is None:
            raise ValueError("Train or load a model before saving it.")

        os.makedirs(save_directory, exist_ok=True)
        self.model.save_pretrained(save_directory)
        self.tokenizer.save_pretrained(save_directory)

        metadata = {
            "base_model_name": self.model_name,
            "max_length": self.max_length,
            "label_all_tokens": self.label_all_tokens,
            "from_pt": self.from_pt,
            "label_list": self.label_list,
        }

        with open(os.path.join(save_directory, "ner_config.json"), "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=True, indent=2)

    @classmethod
    def load(cls, save_directory):
        AutoTokenizer, TFAutoModelForTokenClassification, _ = _require_transformers()

        config_path = os.path.join(save_directory, "ner_config.json")
        with open(config_path, encoding="utf-8") as f:
            metadata = json.load(f)

        instance = cls(
            model_name=metadata.get("base_model_name", save_directory),
            max_length=metadata.get("max_length", 512),
            label_all_tokens=metadata.get("label_all_tokens", False),
            from_pt=metadata.get("from_pt", False),
        )

        instance.label_list = metadata["label_list"]
        instance.label2id = {label: idx for idx, label in enumerate(instance.label_list)}
        instance.id2label = {idx: label for label, idx in instance.label2id.items()}
        instance.tokenizer = AutoTokenizer.from_pretrained(save_directory, use_fast=True)
        instance.model = TFAutoModelForTokenClassification.from_pretrained(save_directory)

        return instance


def output_entities(data, preds, outfile):
    with open(outfile, "w", encoding="utf-8") as outf:
        for sid, tags in zip(data.sentence_ids(), preds):
            inside = False
            sentence = data.get_sentence(sid)

            for token, tag in zip(sentence, tags):
                if tag.startswith("B-"):
                    if inside:
                        print(
                            sid,
                            f"{entity_start}-{entity_end}",
                            entity_form,
                            entity_type,
                            sep="|",
                            file=outf,
                        )

                    entity_form = token["form"]
                    entity_start = token["start"]
                    entity_end = token["end"]
                    entity_type = tag[2:]
                    inside = True

                elif tag.startswith("I-") and inside:
                    entity_form += " " + token["form"]
                    entity_end = token["end"]

                elif inside:
                    print(
                        sid,
                        f"{entity_start}-{entity_end}",
                        entity_form,
                        entity_type,
                        sep="|",
                        file=outf,
                    )
                    inside = False

            if inside:
                print(
                    sid,
                    f"{entity_start}-{entity_end}",
                    entity_form,
                    entity_type,
                    sep="|",
                    file=outf,
                )


def evaluate_predictions(datadir, outfile):
    import evaluator

    evaluator.evaluate("NER", datadir, outfile)
