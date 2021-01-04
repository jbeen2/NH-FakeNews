class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b


class InputFeatures(object):
    """A single set of features of data."""
    
    def __init__(self, input_ids, input_mask, segment_ids, sentence_id=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""
    
    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

class ColaProcessor(DataProcessor):
    """Processor for the CoLA data set (GLUE version)."""
    def __init__(self, title_column, content_column):
        self.title_column = title_column
        self. content_column = content_column
  
    def get_test_examples(self, data):
        """See base class."""
        return self._create_examples(data, "test")

    def _create_examples(self, dataframe, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for idx, row in dataframe.iterrows():
            guid = "%s-%s" % (set_type, idx)
            text_a = row[self.title_column]
            text_b = row[self.content_column]
            examples.append(
              InputExample(guid=guid, text_a=text_a, text_b=text_b))
        return examples

def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer, vocab2id):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label : i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = ""
        tokens_b = ""

        tokens_a = example.text_a    
        tokens_b = example.text_b


        if example.text_b:
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)

        if len(tokens_a) > (max_seq_length - 2):
          tokens_a = tokens_a[:max_seq_length - 2]

        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = [vocab2id[i] if i in vocab2id else vocab2id['[UNK]'] for i in tokens] 

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding
 
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        senid = example.guid+""
        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids))
    return features
    
def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

