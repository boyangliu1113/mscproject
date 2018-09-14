#! /usr/bin/env python

import csv
import multiprocessing
import re

from text_pipeline import pipeline


def cleantext(s=''):
    s = s.replace('\n', ' ')
    s = s.replace('\t', ' ')
    s = s.replace('\r', ' ')
    s = re.sub(r"[\x00-\x1F\x7F]+", "", s)
    s = re.sub(r'^\s*[a-z0-9]\.?\)', "", s)
    s = re.sub(r'\s+[a-z0-9]\.?\)', "", s)
    s = re.sub(r'( [0-9,\.]+)', r"\1 ", s)
    return s


def readcsv(file_name):
    list_csv = []
    with open(file_name, "r") as f:
        qacsv = csv.reader(f)
        for row in qacsv:
            if row[7].strip():
                list_csv.append([cleantext(row[7]), cleantext(row[8]), row[9]])
    return list_csv[1:]


def writetsv(file_name, data, ):
    with open(file_name, "w") as f:
        for row in data:
            s = ''
            for r in row:
                s = s + '\t' + str(r)
            s = s[1:] + '\n'
            f.write(s)


def trainTsv(file_name, data, columns):
    with open(file_name, "w") as f:
        for row in data:
            f.write("{}\t{}\t{}\n".format(row[columns[0]], row[columns[1]], row[2]))


def applyPipeline(cell, pipe):
    if cell:
        converted = pipe([cell])
        if isinstance(converted, list) and converted:
            return ' '.join(filter(None, converted))
    return 'None'
    # row.append(' '.join(filter(None, p)))


def run_pipe(row):
    pipe1 = pipeline.Pipeline(
        [('tokenize_words', {}), ('remove_punct', {}), ('spell_check_list', {}), ('stop_words_removal', {})])
    pipe2 = pipeline.Pipeline([('tokenize_words', {}), ('spell_check_list', {})])
    pipe3 = pipeline.Pipeline([('tokenize_words', {}), ('spell_check_list', {}), ('stop_words_removal', {})])
    pipe4 = pipeline.Pipeline([('tokenize_words', {}), ('stop_words_removal', {})])
    if row is not None:
        row.append(applyPipeline(row[0], pipe1))
        row.append(applyPipeline(row[1], pipe1))
        row.append(applyPipeline(row[0], pipe2))
        row.append(applyPipeline(row[1], pipe2))
        row.append(applyPipeline(row[0], pipe3))
        row.append(applyPipeline(row[1], pipe3))
        row.append(applyPipeline(row[0], pipe4))
        row.append(applyPipeline(row[1], pipe4))
    return row


def main():
    qa_data = readcsv("../features/features_built.csv")
    writetsv("qa_pair.tsv", qa_data)
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        results = pool.map(run_pipe, qa_data, 1)
    writetsv('qa_all.tsv', results)
    trainTsv('qa_spell_punct_stopwords.tsv', results, [3, 4])
    trainTsv('qa_spell.tsv', results, [5, 6])
    trainTsv('qa_spell_stopwords.tsv', results, [7, 8])
    trainTsv('qa_stopwords.tsv', results, [9, 10])


def run_pipe_merged(row):
    pipe1 = pipeline.Pipeline(
        [('tokenize_words', {}), ('remove_punct', {}), ('spell_check_list', {}), ('stop_words_removal', {})])
    pipe2 = pipeline.Pipeline([('tokenize_words', {}), ('spell_check_list', {})])
    pipe3 = pipeline.Pipeline([('tokenize_words', {}), ('spell_check_list', {}), ('stop_words_removal', {})])
    pipe4 = pipeline.Pipeline([('tokenize_words', {}), ('stop_words_removal', {})])
    if row is not None:
        row.append(applyPipeline(row[0], pipe1))
        row.append(applyPipeline(row[1], pipe1))
        row.append(applyPipeline(row[0], pipe2))
        row.append(applyPipeline(row[1], pipe2))
        row.append(applyPipeline(row[0], pipe3))
        row.append(applyPipeline(row[1], pipe3))
        row.append(applyPipeline(row[0], pipe4))
        row.append(applyPipeline(row[1], pipe4))
    return row


def merge_qa(qa_data=None):
    if qa_data is None:
        qa_data = readcsv("../features/features_built.csv")
    merged = []
    for row in qa_data:
        if float(row[2]) == 1.0:
            merged.append([row[0], row[0] + ' ' + row[1], row[2]])
        else:
            merged.append([row[0], row[1], row[2]])
    writetsv("qa_merged.tsv", merged)
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        results = pool.map(run_pipe_merged, merged, 1)
    writetsv('qa_merged_all.tsv', results)
    trainTsv('qa_merged_spell_punct_stopwords.tsv', results, [3, 4])
    trainTsv('qa_merged_spell.tsv', results, [5, 6])
    trainTsv('qa_merged_spell_stopwords.tsv', results, [7, 8])
    trainTsv('qa_merged_stopwords.tsv', results, [9, 10])


if __name__ == '__main__':
    # main()
    merge_qa()
    print("Done!")
#
# Train the model
#       --is_char_based false --training_files qapair.tsv --num_epochs 30 --checkpoint_every 20
# Evaluate the model
#       --vocab_filepath runs/1534242697/checkpoints/vocab --model runs/1534242697/checkpoints/model-160
