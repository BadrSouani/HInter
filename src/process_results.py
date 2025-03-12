import argparse
import os

import numpy
import numpy as np

from tools import methods

parser = argparse.ArgumentParser(description='Process values.')
parser.add_argument('--output_path', type=str, default="../output_mutaint/",
                    help="Path to the output folder")
args = parser.parse_args()
output_folder = args.output_path

results_path = output_folder + "llm_original_mutants_testing_stats.csv"

gender_list = ['female_male', 'female_male_job', 'male_female', 'male_female_job']
race_list = ['african_american', 'african_arab', 'african_asian', 'african_european', 'american_african', 'american_arab', 'american_asian',
             'american_european','arab_african','arab_american','arab_asian','arab_european','asian_african','asian_american',
             'asian_arab','asian_european','european_african','european_american','european_arab','european_asian','majority_minority',
             'majority_mixed','minority_majority','minority_mixed','mixed_majority','mixed_minority']
body_list = ['common_disorder', 'common_hair', 'common_uncommon', 'disorder_common', 'old_young', 'uncommon_common', 'young_old', 'hair_common']

dts = ["imdb", "ecthr_a", "ledgar", "scotus", "eurlex"]
bert_dts = ["ecthr_a", "ledgar", "scotus", "eurlex"]
len_labels = [2, 10, 100, 14, 100]
sts = ["train", "test", "validation"]
models = ['llama2', 'gpt']
all_bert_models = ['nlpaueb/legal-bert-base-uncased', 'bert-base-uncased', 'microsoft/deberta-base', 'roberta-base']
all_models = ['llama2', 'gpt', 'nlpaueb/legal-bert-base-uncased', 'bert-base-uncased', 'microsoft/deberta-base', 'roberta-base']
bert_used_model = ['nlpaueb/legal-bert-base-uncased']
mitigation_models = ['llama2', 'gpt', 'nlpaueb/legal-bert-base-uncased']


mitigation_datasets = bert_dts

all_replacements = [
    ['atomic', [
        [[(x,"") for x in body_list], "Body"],
        [[(x,"") for x in race_list], "Race"],
        [[(x,"") for x in gender_list], "Gender"]
        ]
    ],
    ['intersectional', [
        [[(x,y) for x in body_list for y in race_list], "BodyXRace"],
        [[(x,y) for x in body_list for y in gender_list], "BodyXGender"],
        [[(x,y) for x in race_list for y in gender_list], "RaceXGender"]
        ]
    ]
]

atomic_replacements =  [
        [[(x,"") for x in body_list], "Body"],
        [[(x,"") for x in race_list], "Race"],
        [[(x,"") for x in gender_list], "Gender"]
        ]

intersectional_replacements = [
         [[(x,y) for x in body_list for y in race_list], "Body", "Race"],
         [[(x,y) for x in body_list for y in gender_list], "Body", "Gender"],
         [[(x,y) for x in race_list for y in gender_list], "Race", "Gender"]
 ]


atomic_intersectional_relation = [
    ["Body", "Race", "BodyXRace"],
    ["Body", "Gender", "BodyXGender"],
    ["Race", "Gender", "RaceXGender"]
]

dict_results = dict()
dict_originals = dict()
dict_model_labels = dict()
dict_detail_response = dict()

dict_bert_stats_by_md = dict()
####################Originals scores and file writting###########################
for m in models:
    dict_detail_response[m] = dict()
    dict_originals[m] = dict()
    dict_model_labels[m] = dict()
    for d in dts:
            dict_detail_response[m][d] = dict()
            print("Original of ", m, dts.index(d)+1, "/", len(dts))
            dict_originals[m][d] = dict()
            dict_model_labels[m][d] = dict()
            dict_used = dict_originals[m][d]
            nb_label = len_labels[dts.index(d)]
            range_labels = set(list(range(nb_label)))
            for x in ['time (s)', 'Right', 'Wrong', 'Original'] : dict_used[x] = 0
            for s in (sts if d != "imdb" and d != "winogender" else ["None"]):
                dict_model_labels[m][d][s] = dict()
                dict_detail_response[m][d][s] = dict()
                original_file = "/".join([output_folder, "llm_testing", m, d, "_".join([ s, "original"])]) + ".pkl"
                content = methods.getFromPickle(original_file, "rb")
                dict_o = content[0]
                all_originals = content[1]
                dict_used['Original'] += len(all_originals)
                dict_used["time (s)"] += dict_o["Time (s)"]
                TP = [0]*nb_label
                TN = [0]*nb_label
                FP = [0]*nb_label
                FN = [0]*nb_label
                for o in all_originals:
                    o1 = o[2] if isinstance(o[2], list) else [int(o[2])]
                    o2 = o[3] if isinstance(o[3], list) else [int(o[3])]
                    dict_detail_response[m][d][s][o[1]]=o2
                    actual_labels = set(o1)
                    model_labels = set(o2)
                    dict_model_labels[m][d][s][o[1]] = set(o2)
                    if actual_labels==model_labels: dict_used['Right']+=1
                    else : dict_used['Wrong']+=1
                    for x in actual_labels.intersection(model_labels) : TP[x] += 1
                    for x in range_labels.difference(model_labels).intersection(range_labels.difference(actual_labels)): TN[x] +=1
                    for x in model_labels.difference(actual_labels): FP[x] += 1
                    for x in actual_labels.difference(model_labels): FN[x] += 1

            dict_used['Accuracy'] = dict_used['Right'] / dict_used['Original']
            recall = [TP[x]/(TP[x]+FN[x]) if TP[x]+FN[x] != 0 else 0 for x in range(nb_label)]
            precision = [TP[x]/(TP[x]+FP[x]) if TP[x]+FP[x] != 0 else 0 for x in range(nb_label)]
            macro_recall = numpy.array(recall).sum()/nb_label
            macro_precision = numpy.array(precision).sum()/nb_label
            dict_used['Macro F1 Score'] = (2*macro_precision*macro_recall)/(macro_precision+macro_recall)
            micro_recall = numpy.array(TP).sum()/(numpy.array(TP).sum()+numpy.array(FN).sum())
            micro_precision = numpy.array(TP).sum()/(numpy.array(TP).sum()+numpy.array(FP).sum())
            dict_used['Micro F1 Score'] = (2*micro_precision*micro_recall)/(micro_precision+micro_recall)
            print(m, d, dict_used)
results_lines = [[], ["Original Testing Accuracy/Macro F1 Score/Micro F1 Score"]]
header = ['Original', 'Accuracy', 'Macro F1 Score', "Micro F1 Score"]
top = []

for x in dts : top += [x]*4
results_lines.append([""] + top)
top2 = []
for x in dts : top2 += header
results_lines.append([''] + top2)
for m in models:
    line = [m]
    for d in dts:
        dict_used = dict_originals[m][d]
        line += [dict_used[n] for n in header]
    results_lines.append(line)
###########################################################################



#################### Mutants stats ###########################
name_mutant_folder = "mutants"
print("Processing Mutants ...")
for rep in all_replacements:
    replacement_name = rep[0]
    all_sa = rep[1]
    dict_results[replacement_name] = dict()
    for d in dts:
        print("Mutants ", dts.index(d)+1, "/", len(dts))
        dict_results[replacement_name][d] = dict()
        for tuples, sa in all_sa:
            dict_used = dict_results[replacement_name][d][sa] = dict()
            dict_used["gen_mutants"] = 0
            dict_used["failed_inputs"] = 0
            dict_used["input_generated"] = 0
            dict_used["bad_inputs"] = 0
            dict_used["time (s)"] = 0
            for s in (sts if d != "imdb" and d != "winogender" else ["None"]):
                for x, y in tuples:
                    if d != "winogender":
                        mutant_file = "/".join([output_folder, name_mutant_folder,"_".join(["llm", s, d, s, x+("" if y == "" else "+"+y)+".pkl"])])
                    else : mutant_file = "/".join([output_folder, "mutants","_".join([d, x+("" if y == "" else "+"+y)+".pkl"])])
                    if not os.path.isfile(mutant_file):
                        print(mutant_file)
                        continue
                    content = methods.getFromPickle(mutant_file, "rb")
                    if d != "winogender":
                        dict_m = content[0]
                        all_mutants = content[1]
                    else :
                        dict_m = {content[0][0]:content[0][1], "discarded_test_cases_sentence":0}
                        all_mutants = content[1:]
                    dict_used['gen_mutants'] += len(all_mutants)
                    dict_used["time (s)"] += dict_m["time (s)"]
                    dict_used["bad_inputs"] += dict_m["discarded_test_cases_sentence"]
                    gen_inputs = len([i for i in all_mutants if i[-1] == 1])
                    gen_failed = len([i for i in all_mutants if i[-1] == 0])
                    dict_used['input_generated'] += gen_inputs
                    dict_used['failed_inputs'] += gen_failed

results_lines.append([])
results_lines.append([])
results_lines.append(["Mutants stats"])
top2_header = ["gen_mutants", "input_generated", "failed_inputs", "bad_inputs", "time (s)"]
top = []
for x in dts : top += [x]*len(top2_header)
results_lines.append([""] + top)
top2 = []
for x in dts : top2 += top2_header
results_lines.append([""] + top2)
for rep in all_replacements:
    replacement_name = rep[0]
    all_sa=rep[1]
    for tuples, sa in all_sa:
        line = [sa]
        for d in dts:
            dict_used = dict_results[replacement_name][d][sa]
            line += [dict_used[x] for x in top2_header]
        results_lines.append(line)


#################### Testing stats and file writting and Mitigation start###########################
dict_mitigation = dict()
print("Processing Tests ...")
for rep in all_replacements:
    replacement_name = rep[0]
    all_sa = rep[1]
    for m in models:
        if m not in dict_mitigation : dict_mitigation[m] = dict()
        if m not in dict_results : dict_results[m] = dict()
        for tuples, sa in all_sa:
            dict_results[m][sa] = dict()
            dict_used = dict_results[m][sa]
            for x in ["total test cases", "identical", "used test cases", "error", "error rate","time (s)"] : dict_used[x] = 0
            for d in dts:
                if d not in dict_mitigation[m]: dict_mitigation[m][d] = dict()
                if d not in dict_results[m]: dict_results[m][d] = dict()
                for x in ["total test cases", "identical", "used test cases", "error", "error rate", "time (s)"]:
                    if x not in dict_results[m][d]:
                        dict_results[m][d][x] = 0
                for s in (sts if d != "imdb" and d != "winogender" else ["None"]):
                    if s not in dict_mitigation[m][d]: dict_mitigation[m][d][s] = dict()
                    for x, y in tuples:
                        testing_file = "/".join([output_folder, "llm_testing", m, d, "_".join([ s, x+("" if y == "" else "+"+y)+".pkl"])])
                        if not os.path.isfile(testing_file):
                            print("Missing :", testing_file)
                            continue
                        content = methods.getFromPickle(testing_file, "rb")
                        dict_o = content[0]
                        all_tests = content[1]

                        dict_used["identical"] += dict_o['Identical']
                        dict_used["time (s)"] += dict_o['Time (s)']
                        dict_used['total test cases'] += len(all_tests)+dict_o['Identical']
                        dict_used['used test cases'] += len(all_tests)

                        if replacement_name == "intersectional":
                            dict_results[m][d]["identical"] += dict_o['Identical']
                            dict_results[m][d]["time (s)"] += dict_o['Time (s)']
                            dict_results[m][d]['total test cases'] += len(all_tests) + dict_o['Identical']
                            dict_results[m][d]['used test cases'] += len(all_tests)

                        for test in all_tests:
                            case_num = test[1]
                            replacement_words = (test[0][0], test[0][1]) if len(test[0]) == 7 else (test[0][0], test[0][1], test[0][2], test[0][3])
                            if case_num not in dict_mitigation[m][d][s]: dict_mitigation[m][d][s][case_num] = set()
                            # dict_used['truncated'] += test[-1]
                            model_answer = test[2] if isinstance(test[2], list) else [int(test[2])]
                            if set(model_answer) != dict_model_labels[m][d][s][test[1]]:
                                dict_used['error'] += 1
                                if replacement_name == "intersectional" : dict_results[m][d]['error'] +=1
                                error = 1
                            else:
                                error = 0
                            dict_mitigation[m][d][s][case_num].add(((x,y), replacement_words, error, tuple(test[2])))
                if replacement_name == "intersectional" : dict_results[m][d]['error rate'] = dict_results[m][d]['error'] / dict_results[m][d][
                    'used test cases'] if dict_results[m][d][
                                              'used test cases'] != 0 else "None"

            dict_used['error rate'] = dict_used['error']/dict_used['used test cases'] if dict_used['used test cases'] != 0 else "None"


results_lines.append([])
results_lines.append([])
results_lines.append(["Tests stats"])
top2_header = ["total test cases", "identical", "used test cases", "error", "error rate","time (s)"]
top = []
for x in models : top += [x]*len(top2_header)
results_lines.append([""] + top)
top2 = []
for x in models : top2 += top2_header
results_lines.append([""] + top2)
for rep in all_replacements:
    replacement_name = rep[0]
    all_sa=rep[1]
    for tuples, sa in all_sa:
        line = [sa]
        for m in models:
            dict_used = dict_results[m][sa]
            line += [dict_used[x] for x in top2_header]
        results_lines.append(line)
##########################################################################
##########################LOAD BERT DATA#################################

for m in bert_used_model:
    dict_model_labels[m] = dict()
    for d in bert_dts:
        dict_model_labels[m][d] = dict()
        for s in sts:
            dict_model_labels[m][d][s] = dict()
            prediction_file = "/".join(
                [output_folder, d, m, s, "base_prediction.npy"])
            if not os.path.isfile(prediction_file):
                print("Missing bert mutant :", prediction_file)
                continue
            base_prediction = np.load(prediction_file, allow_pickle=True)
            for x in range(len(base_prediction)):
                dict_model_labels[m][d][s][x] = set(base_prediction[x])

bert_stats_header = ["total test cases","used test cases", "error", "error rate", "failed gen input", "failed gen input rate", "failed error", "failed error rate"]
print("Processing BERT cases for mitigation ...")
for rep in all_replacements:
    replacement_name = rep[0]
    all_sa = rep[1]
    dict_bert_stats_by_md[replacement_name] = dict()
    for m in bert_used_model:
        if m not in dict_bert_stats_by_md[replacement_name]: dict_bert_stats_by_md[replacement_name][m] = dict()
        if m not in dict_mitigation : dict_mitigation[m] = dict()
        if m not in dict_results : dict_results[m] = dict()
        for tuples, sa in all_sa:
            if sa not in dict_results[m] : dict_results[m][sa] = dict()
            for d in bert_dts:
                if d not in dict_bert_stats_by_md[replacement_name][m] : dict_bert_stats_by_md[replacement_name][m][d] = dict()
                if d not in dict_mitigation[m]: dict_mitigation[m][d] = dict()
                if d not in dict_results[m] : dict_results[m][d] = dict()
                for x in ["total test cases","used test cases", "error", "error rate", "failed gen input", "failed gen input rate"]:
                    if x not in dict_results[m][d]: dict_results[m][d][x] = 0
                for x in bert_stats_header:
                    if x not in dict_bert_stats_by_md[replacement_name][m][d]: dict_bert_stats_by_md[replacement_name][m][d][x] = 0
                for s in sts :
                    if s not in dict_mitigation[m][d]: dict_mitigation[m][d][s] = dict()
                    for x, y in tuples:
                        #doing errors
                        mutant_file = "/".join([output_folder,d, m, s, "error_details", ("replacement_replacement_atomic" if y== "" else "intersectional_intersectionality")
                                                +"-"+x+("" if y == "" else "+"+y)+".pkl"])
                        if not os.path.isfile(mutant_file):
                            print("Missing bert error :", mutant_file)
                            continue
                        content = methods.getFromPickle(mutant_file, "rb")


                        valid_mutants = [mut for mut in content if mut[-1]==1]
                        dict_bert_stats_by_md[replacement_name][m][d]["error"] += len(valid_mutants)


                        dict_bert_stats_by_md[replacement_name][m][d]["failed error"] += len(content)-len(valid_mutants)
                        for mutant in valid_mutants:
                            if replacement_name == "intersectional" : dict_results[m][d]["error"] += 1
                            case_num = mutant[4] if y == "" else mutant[6]
                            replacement_words = (mutant[0], mutant[1]) if y == "" else (mutant[0], mutant[1], mutant[2], mutant[3])
                            if case_num not in dict_mitigation[m][d][s]: dict_mitigation[m][d][s][case_num] = set()
                            dict_mitigation[m][d][s][case_num].add(((x,y), replacement_words, 1, tuple(mutant[3]) if y == "" else tuple(mutant[5])))

                        # doing mutants
                        mutant_file = "/".join(
                            [output_folder, "mutants", "_".join([d, s, x + ("" if y == "" else "+" + y) + ".pkl"])])
                        if not os.path.isfile(mutant_file):
                            print("Missing bert mutant :", mutant_file)
                            continue
                        content = methods.getFromPickle(mutant_file, "rb")
                        for mut in content :
                            if mut[-1] != 0 and mut[-1] != 1 :
                                print("Missing invariant check in bert mutant")
                        valid_mutants = [mut for mut in content if mut[-1] == 1]
                        dict_bert_stats_by_md[replacement_name][m][d]["total test cases"] += len(content)
                        dict_bert_stats_by_md[replacement_name][m][d]["used test cases"] += len(valid_mutants)
                        dict_bert_stats_by_md[replacement_name][m][d]["failed gen input"] += len(content)-len(valid_mutants)
                        for mutant in valid_mutants:
                            if replacement_name == "intersectional" : dict_results[m][d]["used test cases"] += 1
                            case_num = mutant[3] if y == "" else mutant[5]
                            replacement_words = (mutant[0], mutant[1]) if y == "" else (
                            mutant[0], mutant[1], mutant[2], mutant[3])
                            if case_num not in dict_mitigation[m][d][s]: dict_mitigation[m][d][s][case_num] = set()
                            temp_list_presence = [muts_scop for muts_scop in dict_mitigation[m][d][s][case_num]
                                                  if muts_scop[0] == (x,y) and muts_scop[1] == replacement_words and muts_scop[2]==1]
                            if len(temp_list_presence) == 0 :
                                dict_mitigation[m][d][s][case_num].add(((x, y), replacement_words, 0, tuple(dict_model_labels[m][d][s][case_num])))
                dict_bert_stats_by_md[replacement_name][m][d]["failed error rate"] = \
                    dict_bert_stats_by_md[replacement_name][m][d]["failed error"] / dict_bert_stats_by_md[replacement_name][m][d]["failed gen input"] if dict_bert_stats_by_md[replacement_name][m][d]["failed gen input"] != 0 else "/"

                dict_bert_stats_by_md[replacement_name][m][d]["error rate"] =\
                    dict_bert_stats_by_md[replacement_name][m][d]["error"]/dict_bert_stats_by_md[replacement_name][m][d]["used test cases"] if dict_bert_stats_by_md[replacement_name][m][d]["used test cases"] != 0 else "/"
                dict_bert_stats_by_md[replacement_name][m][d]["failed gen input rate"] = \
                    dict_bert_stats_by_md[replacement_name][m][d]["failed gen input"] / dict_bert_stats_by_md[replacement_name][m][d]["total test cases"] if dict_bert_stats_by_md[replacement_name][m][d]["total test cases"] != 0 else '/'

                if replacement_name == "intersectional" : dict_results[m][d]['error rate'] = dict_results[m][d]['error'] / dict_results[m][d][
                    'used test cases'] if dict_results[m][d]['used test cases'] != 0 else "None"

############## Loading all bert models errors ############################

all_berts_errors = dict()
print("Processing all BERT errors ...")
for rep in all_replacements:
    replacement_name = rep[0]
    all_sa = rep[1]
    for m in all_bert_models:
        if m == bert_used_model[0] : continue
        if m not in dict_results : dict_results[m] = dict()
        if m not in all_berts_errors : all_berts_errors[m] = dict()
        if m not in dict_bert_stats_by_md[replacement_name] : dict_bert_stats_by_md[replacement_name][m] = dict()
        for tuples, sa in all_sa:
            for d in bert_dts:
                if d not in all_berts_errors[m]: all_berts_errors[m][d] = dict()
                if d not in dict_results[m] : dict_results[m][d]= dict()
                if d not in dict_bert_stats_by_md[replacement_name][m] :dict_bert_stats_by_md[replacement_name][m][d] = dict()
                for x in ["used test cases", "error", "error rate"]:
                    if x not in dict_results[m][d] :
                        dict_results[m][d][x] = 0
                for x in bert_stats_header:
                    if x not in dict_bert_stats_by_md[replacement_name][m][d]: dict_bert_stats_by_md[replacement_name][m][d][x] = 0
                for s in sts :
                    if s not in all_berts_errors[m][d]: all_berts_errors[m][d][s] = dict()
                    for x, y in tuples:
                        #doing errors
                        mutant_file = "/".join([output_folder,d, m, s, "error_details", ("replacement_replacement_atomic" if y== "" else "intersectional_intersectionality")
                                                +"-"+x+("" if y == "" else "+"+y)+".pkl"])
                        if not os.path.isfile(mutant_file):
                            print("Missing bert error :", mutant_file)
                            continue
                        content = methods.getFromPickle(mutant_file, "rb")


                        for mut in content :
                            if mut[-1] != 0 and mut[-1] != 1 :
                                print("Missing invariant check in bert error")
                        valid_mutants = [mut for mut in content if mut[-1] == 1]
                        dict_bert_stats_by_md[replacement_name][m][d]["error"] += len(valid_mutants)

                        dict_bert_stats_by_md[replacement_name][m][d]["failed error"] += len(content)-len(valid_mutants)
                        for mutant in valid_mutants:
                            if replacement_name == "intersectional" : dict_results[m][d]["error"] += 1
                            case_num = mutant[4] if y == "" else mutant[6]
                            replacement_words = (mutant[0], mutant[1]) if y == "" else (mutant[0], mutant[1], mutant[2], mutant[3])
                            if case_num not in all_berts_errors[m][d][s]: all_berts_errors[m][d][s][case_num] = set()
                            all_berts_errors[m][d][s][case_num].add(((x,y), replacement_words, 1, tuple(mutant[3]) if y == "" else tuple(mutant[5])))
                dict_bert_stats_by_md[replacement_name][m][d]["error rate"] =\
                    dict_bert_stats_by_md[replacement_name][m][d]["error"]/dict_bert_stats_by_md[replacement_name][bert_used_model[0]][d]["used test cases"] if dict_bert_stats_by_md[replacement_name][bert_used_model[0]][d]["used test cases"] != 0 else "/"
                dict_bert_stats_by_md[replacement_name][m][d]["failed error rate"] = \
                    dict_bert_stats_by_md[replacement_name][m][d]["failed error"] / dict_bert_stats_by_md[replacement_name][m][d]["failed gen input"] if dict_bert_stats_by_md[replacement_name][m][d]["failed gen input"] != 0 else "/"
                if replacement_name == "intersectional" : dict_results[m][d]['error rate'] = dict_results[m][d]['error'] / dict_results[bert_used_model[0]][d][
                    'used test cases'] if dict_results[bert_used_model[0]][d][
                                              'used test cases'] != 0 else "None"

#Used Test cases numbers
models_header = ["used test cases", "error", "error rate"]
results_lines.append([])
results_lines.append([])
results_lines.append(["Used test cases d/m"])
results_lines.append([""]+all_models)
for d in dts:
    line = [d]
    for m in all_models:
        if d not in dict_results[m] :
            line += ["/"]
        else:
            line += [dict_results[m][d]["used test cases"] if m not in all_bert_models else dict_results[bert_used_model[0]][d]["used test cases"]]
    results_lines.append(line)

#Error numbers
models_header = ["used test cases", "error", "error rate"]
results_lines.append([])
results_lines.append([])
results_lines.append(["Errors d/m"])
results_lines.append([""]+all_models)
for d in dts:
    line = [d]
    for m in all_models:
        if d not in dict_results[m] :
            line += ["/"]
        else:
            line += [dict_results[m][d]["error"]]
    results_lines.append(line)

#Error rates
models_header = ["used test cases", "error", "error rate"]
results_lines.append([])
results_lines.append([])
results_lines.append(["Errors rates d/m"])
results_lines.append([""]+all_models)
dict_total_model_error = dict()
for m in all_models:
    dict_total_model_error[m] = dict()
    dict_total_model_error[m]["nb_all_errors"] = 0
    dict_total_model_error[m]["nb_all_used"] = 0
for d in dts:
    line = [d]
    for m in all_models:
        if d not in dict_results[m] :
            line += ["/"]
        else:
            line += [dict_results[m][d]["error"]/(dict_results[m][d]["used test cases"] if m not in all_bert_models else dict_results[bert_used_model[0]][d]["used test cases"])]
            dict_total_model_error[m]["nb_all_errors"] += dict_results[m][d]["error"]
            dict_total_model_error[m]["nb_all_used"] += dict_results[m][d]["used test cases"] if m not in all_bert_models else dict_results[bert_used_model[0]][d]["used test cases"]
    results_lines.append(line)
model_error_rates = [dict_total_model_error[m]["nb_all_errors"]/(dict_total_model_error[m]["nb_all_used"]
                            if m not in all_bert_models else dict_total_model_error[bert_used_model[0]]["nb_all_used"]) for m in all_models]
results_lines.append(["Total"]+model_error_rates)

###################################### Writing all bert stats (gen rate, error rate, false error rate) ###################################
###########################  Bert stats model/dataset  ######################## #

results_lines.append(["Bert Stats"])
line = ["", "", ""]
for m in all_bert_models : line+=[m]*len(bert_stats_header)
results_lines.append(line)
line = ["","", ""]
for m in all_bert_models : line+=bert_stats_header
results_lines.append(line)
for rep in all_replacements:
    replacement_name = rep[0]
    for d in bert_dts:
        line = ["", replacement_name, d]
        for m in all_bert_models:
            for x in bert_stats_header : line += ([dict_bert_stats_by_md[replacement_name][m][d][x]]) if x in ["error", "error rate", "failed error", "failed error rate"] else ([dict_bert_stats_by_md[replacement_name][bert_used_model[0]][d][x]])
        results_lines.append(line)
##################################################################################################################
############## init Mitigation Results ############
dict_mitigation_results = dict()
dict_mitigation_results["input"] = dict()
dict_mitigation_results["model"] = dict()
##########################################################################
########################## Finish Mitigation with BERT ####################################
#####detect if case has at least 1 error and more than 3 mutants

print("Processing Mitigation by Input...")
all_cases_bert_intersectional = set()
all_cases_bert_atomic = set()
bert_model_to_use = 'nlpaueb/legal-bert-base-uncased'
for d in mitigation_datasets:
    for s in sts:
        for cn in dict_mitigation[bert_model_to_use][d][s]:
            for elem in dict_mitigation[bert_model_to_use][d][s][cn]:
                if elem[0][1]=="" : all_cases_bert_atomic.add((d, s, cn))
                else : all_cases_bert_intersectional.add((d, s, cn))


for rep, all_cases in [["atomic", all_cases_bert_atomic], ["intersectional", all_cases_bert_intersectional]]:
    dict_mitigation_results["input"][rep] = dict()
    for m in mitigation_models:
        dict_mitigation_results["input"][rep][m] = dict()
        dict_mitigation_results["input"][rep][m]["error_crossed"] = 0
        dict_mitigation_results["input"][rep][m]["error_fixed"] = 0
        dict_mitigation_results["input"][rep][m]["cases"] = 0
        dict_mitigation_results["input"][rep][m]["cases_w1e"] = 0
        list_errors = []
        list_mutants = []
        list_non_errors = []
        list_errors_with_errors = []
        list_mutants_with_errors = []
        list_non_errors_with_errors = []
        for cases in all_cases:

            case_num = cases[2]
            case_dataset = cases[0]
            case_set = cases[1]

            if case_num not in dict_mitigation[m][case_dataset][case_set]: continue # if model dont have any mutant of this case, continue
            case_model_mutants = list(dict_mitigation[m][case_dataset][case_set][case_num]) # all inputs for that model of that case (with errors)
            nb_mutants = len(case_model_mutants)

            if nb_mutants < 3: continue #cant do majority, skip
            case_model_errors = [case_model_mutants for x in case_model_mutants if x[2] == 1]
            nb_errors = len(case_model_errors)
            nb_non_errors = nb_mutants - nb_errors
            dict_mitigation_results["input"][rep][m]["error_crossed"] += nb_errors
            dict_mitigation_results["input"][rep][m]["cases"] += 1

            list_errors.append(nb_errors)
            list_non_errors.append(nb_non_errors)
            list_mutants.append(nb_mutants)

            if nb_non_errors != 0 :
                list_errors_with_errors.append(nb_errors)
                list_non_errors_with_errors.append(nb_non_errors)
                list_mutants_with_errors.append(nb_mutants)
                dict_mitigation_results["input"][rep][m]["cases_w1e"] += 1
                if nb_non_errors > nb_errors : dict_mitigation_results["input"][rep][m]["error_fixed"] += 1

        dict_mitigation_results["input"][rep][m]["avg_mutant"] = np.mean(list_mutants)
        dict_mitigation_results["input"][rep][m]["avg_error"] = np.mean(list_errors)
        dict_mitigation_results["input"][rep][m]["avg_non_error"] = np.mean(list_non_errors)
        dict_mitigation_results["input"][rep][m]["avg_mutant_w1e"] = np.mean(list_mutants_with_errors)
        dict_mitigation_results["input"][rep][m]["avg_error_w1e"] = np.mean(list_errors_with_errors)
        dict_mitigation_results["input"][rep][m]["avg_non_error_w1e"] = np.mean(list_non_errors_with_errors)
        dict_mitigation_results["input"][rep][m]["error_fix_rate"] = dict_mitigation_results["input"][rep][m]["error_fixed"] / dict_mitigation_results["input"][rep][m]["error_crossed"]

results_lines.append([])
results_lines.append([])
results_lines.append(["Input Mitigation"])
top2_header = ["error_crossed", "error_fixed","error_fix_rate", "cases", "avg_mutant", "avg_error", "avg_non_error",
               "cases_w1e","avg_mutant_w1e", "avg_error_w1e", "avg_non_error_w1e"]
top1 = []
for x in mitigation_models : top1 += [x]*len(top2_header)
results_lines.append([""] + top1)
top2 = []
for x in mitigation_models : top2 += top2_header
results_lines.append([""] + top2)

for rep in ["atomic", "intersectional"]:
    line = [rep]
    for m in mitigation_models:
        line += [dict_mitigation_results["input"][rep][m][x] for x in top2_header]
    results_lines.append(line)

################################################################################
######################################## Model Mitigation ########################################
#####detect if mutant has at least 1 error and test in more than 3 models
print("Processing Mitigation by Model...")
all_inputs_atomic = []
all_inputs_intersectional = []

for d in mitigation_datasets:
    for s in sts:
        for cn in dict_mitigation[bert_model_to_use][d][s]:
            if not cn in dict_mitigation["llama2"][d][s] or not cn in dict_mitigation["gpt"][d][s] : continue
            for elem in dict_mitigation[bert_model_to_use][d][s][cn]:


                temp_list_presence = [x for x in dict_mitigation["llama2"][d][s][cn] if x[0]==elem[0] and x[1]==elem[1]]
                if len(temp_list_presence) == 0 : continue
                if len(temp_list_presence) > 1:
                    print("ERROR TOO MANY CANDIDATES !!!")
                    continue
                answer1 = [temp_list_presence[0][2], temp_list_presence[0][3], tuple(dict_model_labels["llama2"][d][s][cn])]

                temp_list_presence = [x for x in dict_mitigation["gpt"][d][s][cn] if
                                      x[0] == elem[0] and x[1] == elem[1]]
                if len(temp_list_presence) == 0: continue
                if len(temp_list_presence) > 1:
                    print("ERROR TOO MANY CANDIDATES !!!")
                    continue
                answer2 = [temp_list_presence[0][2], temp_list_presence[0][3], tuple(dict_model_labels["gpt"][d][s][cn])]

                elemns_to_keep = [elem[0], elem[1], [elem[-2], elem[-1], tuple(dict_model_labels[bert_model_to_use][d][s][cn])]
                                                                   , answer1, answer2]
                if elem[0][1] == "" : all_inputs_atomic.append(elemns_to_keep)
                else : all_inputs_intersectional.append(elemns_to_keep)


for rep, all_inputs in [["atomic", all_inputs_atomic], ["intersectional", all_inputs_intersectional]]:
    dict_mitigation_results["model"][rep] = dict()
    dict_mitigation_results["model"][rep]["BL"] = 0
    dict_mitigation_results["model"][rep]["LG"] = 0
    dict_mitigation_results["model"][rep]["BG"] = 0
    dict_mitigation_results["model"][rep]["BLG"] = 0
    dict_mitigation_results["model"][rep]["mutant"] = 0
    dict_mitigation_results["model"][rep]["BL_w1e"] = 0
    dict_mitigation_results["model"][rep]["LG_w1e"] = 0
    dict_mitigation_results["model"][rep]["BG_w1e"] = 0
    dict_mitigation_results["model"][rep]["BLG_w1e"] = 0
    dict_mitigation_results["model"][rep]["mutant_w1e"] = 0

    for m in mitigation_models:
        dict_mitigation_results["model"][rep][m] = dict()
        dict_mitigation_results["model"][rep][m]["error_crossed"] = 0
        dict_mitigation_results["model"][rep][m]["error_fixed"] = 0

    for input in all_inputs:
        error_b = input[2][0]
        error_l = input[3][0]
        error_g = input[4][0]
        actual_answer_b = tuple(input[2][2])
        actual_answer_l = tuple(input[3][2])
        actual_answer_g = tuple(input[4][2])
        rb = tuple(input[2][1])
        rl = tuple(input[3][1])
        rg = tuple(input[4][1])

        dict_mitigation_results["model"][rep]["mutant"] += 1
        if rb == rl and rb == rg:
            dict_mitigation_results["model"][rep]["BLG"] += 1

        else:
            if rb == rl :
                dict_mitigation_results["model"][rep]["BL"] += 1

            if rg == rl:
                dict_mitigation_results["model"][rep]["LG"] += 1

            if rb == rg:
                dict_mitigation_results["model"][rep]["BG"] += 1


        if error_b + error_l + error_g == 0 : continue #No error to fix

        dict_mitigation_results["model"][rep]["mutant_w1e"] += 1
        if rb == rl and rb == rg: dict_mitigation_results["model"][rep]["BLG_w1e"] += 1
        else:
            if rb == rl : dict_mitigation_results["model"][rep]["BL_w1e"] += 1
            if rg == rl: dict_mitigation_results["model"][rep]["LG_w1e"] += 1
            if rb == rg: dict_mitigation_results["model"][rep]["BG_w1e"] += 1


        if error_b :
            dict_mitigation_results["model"][rep][bert_model_to_use]["error_crossed"] += 1
            if actual_answer_b == rl and actual_answer_b == rg :
                dict_mitigation_results["model"][rep][bert_model_to_use]["error_fixed"] += 1

        if error_l :
            dict_mitigation_results["model"][rep]["llama2"]["error_crossed"] += 1
            if actual_answer_l == rb and actual_answer_l == rg :
                dict_mitigation_results["model"][rep]["llama2"]["error_fixed"] += 1

        if error_g :
            dict_mitigation_results["model"][rep]["gpt"]["error_crossed"] += 1
            if actual_answer_g == rb and actual_answer_g == rl :
                dict_mitigation_results["model"][rep]["gpt"]["error_fixed"] += 1

    dict_mitigation_results["model"][rep]["gpt"]["error_fix_rate"] = \
        dict_mitigation_results["model"][rep]["gpt"]["error_fixed"] / dict_mitigation_results["model"][rep]["gpt"]["error_crossed"]

    dict_mitigation_results["model"][rep][bert_model_to_use]["error_fix_rate"] = \
        dict_mitigation_results["model"][rep][bert_model_to_use]["error_fixed"] / dict_mitigation_results["model"][rep][bert_model_to_use]["error_crossed"]

    dict_mitigation_results["model"][rep]["llama2"]["error_fix_rate"] = \
        dict_mitigation_results["model"][rep]["llama2"]["error_fixed"] / dict_mitigation_results["model"][rep]["llama2"]["error_crossed"]


results_lines.append([])
results_lines.append([])
results_lines.append(["Model Mitigation"])
top2_header = ["error_crossed", "error_fixed","error_fix_rate"]
top1 = []
for x in mitigation_models : top1 += [x]*len(top2_header)
results_lines.append([""] + top1)
top2 = []
for x in mitigation_models : top2 += top2_header
results_lines.append([""] + top2)

for rep in ["atomic", "intersectional"]:
    line = [rep]
    for m in mitigation_models:
        line += [dict_mitigation_results["model"][rep][m][x] for x in top2_header]
    results_lines.append(line)

results_lines.append([])
results_lines.append([])
results_lines.append(["Model Mitigation Answer similarity"])
top2_header = ["mutant", "BL", "LG", "BG", "BLG", "mutant_w1e", "BL_w1e", "LG_w1e", "BG_w1e", "BLG_w1e"]
results_lines.append([""] + top2_header)

for rep in ["atomic", "intersectional"]:
    line = [rep]
    line += [dict_mitigation_results["model"][rep][x] for x in top2_header]
    results_lines.append(line)


###############################################
dict_mitigation_results["model"]["L"] = 0
dict_mitigation_results["model"]["G"] = 0
dict_mitigation_results["model"]["B"] = 0
dict_mitigation_results["model"]["LG"] = 0
dict_mitigation_results["model"]["LB"] = 0
dict_mitigation_results["model"]["GB"] = 0
dict_mitigation_results["model"]["LGB"] = 0

for input in all_inputs_atomic+all_inputs_intersectional:
    error_b = input[2][0]
    error_l = input[3][0]
    error_g = input[4][0]
    if error_b and error_l and error_g : dict_mitigation_results["model"]["LGB"] +=1
    elif error_b and error_l : dict_mitigation_results["model"]["LB"] += 1
    elif error_g and error_l: dict_mitigation_results["model"]["LG"] += 1
    elif error_b and error_g:dict_mitigation_results["model"]["GB"] += 1
    elif error_b :dict_mitigation_results["model"]["B"] += 1
    elif error_l:dict_mitigation_results["model"]["L"] += 1
    elif error_g: dict_mitigation_results["model"]["G"] += 1

results_lines.append([])
results_lines.append([])
results_lines.append(["Common errors"])
top2_header = ["L", "G", "B", "LG", "LB", "GB", "LGB"]
results_lines.append([""]+top2_header)
line = ["Errors"]
line += [dict_mitigation_results["model"][x] for x in top2_header]
results_lines.append(line)



##########################################################################
#########################      Truth Table     ###########################
presence = [
    ["000",[0, 0, 0]],
    ["010",[0, 1, 0]],
    ["001",[0, 0, 1]],
    ["011",[0, 1, 1]],
    ["100",[1, 0, 0]],
    ["110",[1, 1, 0]],
    ["101",[1, 0, 1]],
    ["111",[1, 1, 1]]
]

print("Doing truth tables ...")
dict_truth_table = dict()
dict_distinct_truth_table = dict()
dict_presence = dict()
all_table_elems = ["Intersectional test cases", "Test cases used"]
fix_table = [
             "Total Unique Atomic Bias",
             "Total Unique Intersectional Bias",
             "Total Unique Bias Inducing Bias",
             "Total Unique Hidden Intersectional Bias",
             "Total Unique Atomic share Intersectional Bias"]
cpt = 0
for x in fix_table : dict_distinct_truth_table[x] = set()
for m in mitigation_models:
    dict_presence[m] = dict()
    dict_truth_table[m] = dict()
    for all_list_pairs, sa1, sa2 in intersectional_replacements:

        intersectional_name = sa1+"X"+sa2
        dict_truth_table[m][intersectional_name] = dict()
        for ate in all_table_elems:dict_truth_table[m][intersectional_name][ate] = 0
        for pn, p in presence :
            dict_truth_table[m][intersectional_name][pn] = 0
        for d in dts:
            if d not in dict_presence[m] : dict_presence[m][d] = dict()
            if d not in dict_mitigation[m]: continue
            for s in (sts if d != "imdb" and d != "winogender" else ["None"]):
                if s not in dict_presence[m][d] : dict_presence[m][d][s] = dict()
                for x, y in all_list_pairs:
                    if x not in dict_presence[m][d][s] : dict_presence[m][d][s][x] = dict()
                    if y not in dict_presence[m][d][s]: dict_presence[m][d][s][y] = dict()
                    for cn in dict_mitigation[m][d][s]:
                        intersectional_mutants = [e for e in dict_mitigation[m][d][s][cn] if e[0][0] == x and e[0][1] == y]
                        if len(intersectional_mutants) == 0 : continue
                        for intersectional_mutant in intersectional_mutants:
                            dict_truth_table[m][intersectional_name]["Intersectional test cases"] += 1
                            p1 = (intersectional_mutant[1][0], intersectional_mutant[1][1])
                            p2 = (intersectional_mutant[1][2], intersectional_mutant[1][3])
                            if cn not in dict_presence[m][d][s][x]:
                                atomic_1_mutants = [f for f in dict_mitigation[m][d][s][cn] if len(f[1])==2 and f[0][0] == x and f[1][0] == p1[0] and f[1][1] == p1[1]]
                                if len(atomic_1_mutants) == 0 : continue
                                dict_presence[m][d][s][x][cn] = atomic_1_mutants[0][2]
                            if cn not in dict_presence[m][d][s][y]:
                                atomic_2_mutants = [g for g in dict_mitigation[m][d][s][cn] if len(g[1])==2 and g[0][0] == y and g[1][0] == p2[0] and g[1][1] == p2[1]]
                                if len(atomic_2_mutants) == 0: continue
                                dict_presence[m][d][s][y][cn] = atomic_2_mutants[0][2]
                            cpt += 1 ###################
                            dict_truth_table[m][intersectional_name]["Test cases used"] += 1
                            error0 = intersectional_mutant[2]
                            error1 = dict_presence[m][d][s][x][cn]
                            error2 = dict_presence[m][d][s][y][cn]

                            elem_set0 = (m, d, s, x, y, cn, p1,p2)
                            elem_set1 = (m, d, s, x, cn, p1)
                            elem_set2 = (m, d, s, y, cn, p2)

                            if error1:
                                dict_distinct_truth_table["Total Unique Atomic Bias"].add(elem_set1)
                            if error2:
                                dict_distinct_truth_table["Total Unique Atomic Bias"].add(elem_set2)

                            if error0:
                                dict_distinct_truth_table["Total Unique Intersectional Bias"].add(elem_set0)
                                if error1 : dict_distinct_truth_table["Total Unique Atomic share Intersectional Bias"].add(elem_set1)
                                if error2 : dict_distinct_truth_table["Total Unique Atomic share Intersectional Bias"].add(elem_set2)
                                if not error1 and not error2 ==0 : dict_distinct_truth_table["Total Unique Hidden Intersectional Bias"].add(elem_set0)

                            input_presence = [error0, error1, error2]
                            for pn, p in presence:
                                if p == input_presence :
                                    dict_truth_table[m][intersectional_name][pn] += 1
                                    break

print("cpt :", cpt)
dict_distinct_truth_table["Total Unique Bias Inducing Mutants"] = \
    dict_distinct_truth_table["Total Unique Intersectional Bias"].union(
dict_distinct_truth_table["Total Unique Atomic Bias"]
    )

results_lines.append([])
results_lines.append([])
results_lines.append(["Unique Truth Table "])
top_header = [""] + [x for x in fix_table]
results_lines.append(top_header)
line = [""]+[len(dict_distinct_truth_table[x]) for x in fix_table]
results_lines.append(line)

for m in mitigation_models:
    results_lines.append([])
    results_lines.append([])
    results_lines.append(["Truth Table "+m])
    top_header = [""]
    for all_list_pairs, sa1, sa2 in intersectional_replacements:
        top_header += ["Intersectional "+sa1+"X"+sa2, "Atomic "+sa1, "Atomic "+sa2, "Quantity", "Rate"] + all_table_elems
    results_lines.append(top_header)

    for pn, p in presence:
        line = [""]
        for all_list_pairs, sa1, sa2 in intersectional_replacements:
            intersectional_name = sa1 + "X" + sa2
            rate = round(dict_truth_table[m][intersectional_name][pn]/dict_truth_table[m][intersectional_name]["Test cases used"],2)
            line += ["E" if x=="1" else "B" for x in pn]
            line += [dict_truth_table[m][intersectional_name][pn]]
            line += [rate]
            line += [dict_truth_table[m][intersectional_name]["Intersectional test cases"]]
            line += [dict_truth_table[m][intersectional_name]["Test cases used"]]
        results_lines.append(line)

##########################################################################

##########################################################################
#########################      Truth Table But distinct     ###########################

for m in mitigation_models:
    results_lines.append([])
    results_lines.append([])
    results_lines.append(["Truth Table "+m])
    top_header = [""]
    for all_list_pairs, sa1, sa2 in intersectional_replacements:
        top_header += ["Intersectional "+sa1+"X"+sa2, "Atomic "+sa1, "Atomic "+sa2, "Quantity", "Rate"] + all_table_elems
    results_lines.append(top_header)

    for pn, p in presence:
        line = [""]
        for all_list_pairs, sa1, sa2 in intersectional_replacements:
            intersectional_name = sa1 + "X" + sa2
            rate = round(dict_truth_table[m][intersectional_name][pn]/dict_truth_table[m][intersectional_name]["Test cases used"],2)
            line += ["E" if x=="1" else "B" for x in pn]
            line += [dict_truth_table[m][intersectional_name][pn]]
            line += [rate]
            line += [dict_truth_table[m][intersectional_name]["Intersectional test cases"]]
            line += [dict_truth_table[m][intersectional_name]["Test cases used"]]
        results_lines.append(line)


########################Distinct Cases###############################

distinct_cases = dict()
distinct_cases_w1e = dict()

distinct_cases_atomic = dict()
distinct_cases_w1e_atomic = dict()

distinct_cases_inter = dict()
distinct_cases_w1e_inter = dict()

total_distinct_cases_atomic = set()
total_distinct_cases_w1e_atomic = set()

total_distinct_cases_inter = set()
total_distinct_cases_w1e_inter = set()
for m in mitigation_models:
    distinct_cases[m] =  dict()
    distinct_cases_w1e[m] = dict()

    distinct_cases_atomic[m] = dict()
    distinct_cases_w1e_atomic[m] = dict()

    distinct_cases_inter[m] = dict()
    distinct_cases_w1e_inter[m] = dict()
    for d in dts :
        if d not in dict_mitigation[m]: continue
        distinct_cases[m][d] = set()
        distinct_cases_w1e[m][d] = set()

        distinct_cases_atomic[m][d] = set()
        distinct_cases_w1e_atomic[m][d] = set()

        distinct_cases_inter[m][d] = set()
        distinct_cases_w1e_inter[m][d] = set()
        for s in (sts if d != "imdb" and d != "winogender" else ["None"]):
            for cn in dict_mitigation[m][d][s]:
                distinct_cases[m][d].add((s, cn))
                temp_list_presence_atomic = len([x for x in dict_mitigation[m][d][s][cn] if x[0][1] == ""])
                temp_list_presence_inter = len([x for x in dict_mitigation[m][d][s][cn] if x[0][1] != ""])
                if temp_list_presence_atomic>0 :
                    distinct_cases_atomic[m][d].add((s, cn))
                    total_distinct_cases_atomic.add((d, s, cn))
                if temp_list_presence_inter>0:
                    distinct_cases_inter[m][d].add((s, cn))
                    total_distinct_cases_inter.add((d, s, cn))

                temp_list_presence_any_error = len([x for x in dict_mitigation[m][d][s][cn] if x[2] == 1])
                temp_list_presence_atomic_error = len([x for x in dict_mitigation[m][d][s][cn] if x[2] ==1 and  x[0][1] == ""])
                temp_list_presence_inter_error = len([x for x in dict_mitigation[m][d][s][cn] if x[2] ==1 and  x[0][1] != ""])
                if temp_list_presence_any_error>0 : distinct_cases_w1e[m][d].add((s, cn))
                if temp_list_presence_atomic_error > 0:
                    distinct_cases_w1e_atomic[m][d].add((s, cn))
                    total_distinct_cases_w1e_atomic.add((d, s, cn))
                if temp_list_presence_inter_error > 0:
                    distinct_cases_w1e_inter[m][d].add((s, cn))
                    total_distinct_cases_w1e_inter.add((d, s, cn))

for m in all_bert_models:
    if m not in all_berts_errors : continue
    if m not in distinct_cases_w1e: distinct_cases_w1e[m]= dict()
    if m not in distinct_cases_w1e_atomic: distinct_cases_w1e_atomic[m] = dict()
    if m not in distinct_cases_w1e_inter: distinct_cases_w1e_inter[m] = dict()
    for d in dts :
        if d not in all_berts_errors[m]: continue
        if d not in distinct_cases_w1e[m]: distinct_cases_w1e[m][d] = set()
        if d not in distinct_cases_w1e_atomic[m]: distinct_cases_w1e_atomic[m][d] = set()
        if d not in distinct_cases_w1e_inter[m]: distinct_cases_w1e_inter[m][d] = set()
        for s in (sts if d != "imdb" and d != "winogender" else ["None"]):
            for cn in all_berts_errors[m][d][s]:
                temp_list_presence_any_error = len([x for x in all_berts_errors[m][d][s][cn] if x[2] == 1])
                temp_list_presence_atomic_error = len([x for x in all_berts_errors[m][d][s][cn] if x[2] ==1 and  x[0][1] == ""])
                temp_list_presence_inter_error = len([x for x in all_berts_errors[m][d][s][cn] if x[2] ==1 and  x[0][1] != ""])
                if temp_list_presence_any_error>0 : distinct_cases_w1e[m][d].add((s, cn))
                if temp_list_presence_atomic_error > 0:
                    distinct_cases_w1e_atomic[m][d].add((s, cn))
                    total_distinct_cases_w1e_atomic.add((d, s, cn))

                if temp_list_presence_inter_error > 0:
                    distinct_cases_w1e_inter[m][d].add((s, cn))
                    total_distinct_cases_w1e_inter.add((d, s, cn))

# for title_name in ["Atomic", "Intersectional", "Atomic and Intersectional"]
for name, dc, dcw1e in [["Atomic", distinct_cases_atomic, distinct_cases_w1e_atomic],
                        ["Intersectional", distinct_cases_inter, distinct_cases_w1e_inter],
                        ["Total", distinct_cases, distinct_cases_w1e]]:
    for dic, title in [[dc, "Distinct Cases used to create Mutants"], [dcw1e, "Distinct Cases used to create Mutants w1e"]] :
        results_lines.append([])
        results_lines.append([])
        results_lines.append([name])
        results_lines.append([title])
        results_lines.append([""]+models+all_bert_models)
        for d in dts:
            line = [d]
            for m in models+all_bert_models:
                line += [len(dic[m][d]) if m in dic and d in dic[m] else "/"]
            results_lines.append(line)
results_lines.append([["Total distinct cases atomic"],[ len(total_distinct_cases_atomic) ]])
results_lines.append([["Total distinct cases atomic w1e"],[ len(total_distinct_cases_w1e_atomic) ]])
results_lines.append([["Total distinct cases intersectional"],[ len(total_distinct_cases_inter) ]])
results_lines.append([["Total distinct cases intersectional w1e"],[ len(total_distinct_cases_w1e_inter) ]])
##########################################################################
########################Distinct Originals################################

all_mutants_errors = dict_mitigation.copy()
all_mutants_errors.update(all_berts_errors)
all_used_originals = set()

dict_originals_has_error = dict()
print("Doing Distinct Originals")
for rep in all_replacements:
    replacement_name = rep[0]
    all_sa = rep[1]
    for tuples, sa in all_sa:
        dict_originals_has_error[sa] = dict()
        for m in all_models:
            dict_originals_has_error[sa][m] = dict()
            for d in dts:
                if d not in all_mutants_errors[m]: continue
                dict_originals_has_error[sa][m][d] = dict()
                for s in (sts if d != "imdb" and d != "winogender" else ["None"]):
                    dict_originals_has_error[sa][m][d][s] = dict()
                    for cn in all_mutants_errors[m][d][s]:
                        if cn in dict_originals_has_error[sa][m][d][s] and dict_originals_has_error[sa][m][d][s][cn]==1: continue
                        all_used_originals.add((d, s, cn))
                        for tuple in tuples:
                            for e in all_mutants_errors[m][d][s][cn] :
                                if e[0] == tuple :
                                    if e[2] == 1:
                                        dict_originals_has_error[sa][m][d][s][cn] = 1
                                        break
                                    else: dict_originals_has_error[sa][m][d][s][cn]=0



dict_original_repartition = dict()
originals_columns = ["1", "2", "12", "1-2", "1-12", "2-12", "1-2-12", "No Error", "No Mutant"]
for a1, a2, i in atomic_intersectional_relation:
    dict_original_repartition[i] = dict()
    for x in originals_columns : dict_original_repartition[i][x]=0
    for d,s,cn in all_used_originals:
        in_a1 = None
        in_a2 = None
        in_i = None
        for m in all_models:
            if d in dict_originals_has_error[a1][m] and cn in dict_originals_has_error[a1][m][d][s]:
                if in_a1 == None : in_a1 = False
                if dict_originals_has_error[a1][m][d][s][cn]: in_a1 = True

            if d in dict_originals_has_error[a2][m] and cn in dict_originals_has_error[a2][m][d][s]:
                if in_a2 == None: in_a2 = False
                if dict_originals_has_error[a2][m][d][s][cn]: in_a2 = True

            if d in dict_originals_has_error[i][m] and cn in dict_originals_has_error[i][m][d][s]:
                if in_i == None: in_i = False
                if dict_originals_has_error[i][m][d][s][cn]: in_i = True

        if in_a1 == None or in_a2 == None or in_i == None: dict_original_repartition[i]["No Mutant"] += 1
        elif in_a1==True and in_a2==True and in_i==True : dict_original_repartition[i]["1-2-12"]+=1
        elif in_a2==True and in_i==True : dict_original_repartition[i]["2-12"]+=1
        elif in_a1==True and in_i==True : dict_original_repartition[i]["1-12"]+=1
        elif in_a1==True and in_a2==True : dict_original_repartition[i]["1-2"]+=1
        elif in_i==True: dict_original_repartition[i]["12"]+=1
        elif in_a2==True: dict_original_repartition[i]["2"]+=1
        elif in_a1==True: dict_original_repartition[i]["1"]+=1
        else : dict_original_repartition[i]["No Error"]+=1

results_lines.append([])
results_lines.append([])
results_lines.append(["Distinct Originals"])
results_lines.append([" "]+originals_columns)
for a1, a2, i in atomic_intersectional_relation:
    results_lines.append([i]+[dict_original_repartition[i][x] for x in originals_columns])

#####################################################################"
#########################      Hidden errors for all models     ###########################
print("Doing Hidden errors...")
hidden_values = ["Number Errors", "Hidden Errors", "Hidden Error rate"]
dict_hidden_error = dict()

hidden_ones = []
for m in all_models:
    dict_hidden_error[m] = dict()
    if m not in dict_presence : dict_presence[m] = dict()
    mu = bert_used_model[0] if m != bert_used_model[0] and m in all_bert_models else None
    for d in dts:
        dict_hidden_error[m][d] = dict()
        if d not in dict_presence[m]: dict_presence[m][d] = dict()
        for x in hidden_values: dict_hidden_error[m][d][x] = 0
        if d not in all_mutants_errors[m]: continue
        for all_list_pairs, sa1, sa2 in intersectional_replacements:
            intersectional_name = sa1+"X"+sa2
            for s in (sts if d != "imdb" and d != "winogender" else ["None"]):
                if s not in dict_presence[m][d] : dict_presence[m][d][s] = dict()
                for x, y in all_list_pairs:
                    if x not in dict_presence[m][d][s] : dict_presence[m][d][s][x] = dict()
                    if y not in dict_presence[m][d][s]: dict_presence[m][d][s][y] = dict()
                    for cn in all_mutants_errors[m][d][s]:
                        tuple_focused = (x,y)
                        intersectional_mutants = [e for e in all_mutants_errors[m][d][s][cn] if e[0] == tuple_focused]
                        if len(intersectional_mutants) == 0 : continue
                        for intersectional_mutant in intersectional_mutants:

                            l1 = (x,"")
                            p1 = (intersectional_mutant[1][0], intersectional_mutant[1][1])
                            l2 = (y, "")
                            p2 = (intersectional_mutant[1][2], intersectional_mutant[1][3])
                            if cn not in dict_presence[m][d][s][x]:
                                atomic_1_mutants = [e for e in all_mutants_errors[m][d][s][cn] if
                                                    e[0] == l1 and e[1] == p1]
                                if mu == None:

                                    if len(atomic_1_mutants) == 0 : continue
                                    dict_presence[m][d][s][x][cn] = atomic_1_mutants[0][2]
                                else:
                                    if len(atomic_1_mutants) != 0 :
                                        dict_presence[m][d][s][x][cn] = 1
                                    else:
                                        if cn not in all_mutants_errors[mu][d][s] :
                                            print(cn, " from ", m)
                                            continue
                                        atomic_1_mutants = [e for e in all_mutants_errors[mu][d][s][cn] if
                                                            e[0] == l1 and e[1] == p1]
                                        if len(atomic_1_mutants)==0 : continue
                                        dict_presence[m][d][s][x][cn] = 0

                            if cn not in dict_presence[m][d][s][y]:
                                atomic_2_mutants = [e for e in all_mutants_errors[m][d][s][cn] if
                                                    e[0] == l2 and e[1] == p2]
                                if mu == None:

                                    if len(atomic_2_mutants) == 0 : continue
                                    dict_presence[m][d][s][y][cn] = atomic_2_mutants[0][2]
                                else:
                                    if len(atomic_2_mutants) != 0 :
                                        dict_presence[m][d][s][y][cn] = 1
                                    else :
                                        if cn not in all_mutants_errors[mu][d][s] :
                                            print(cn, " from ", m)
                                            continue
                                        atomic_2_mutants = [e for e in all_mutants_errors[mu][d][s][cn] if
                                                            e[0] == l2 and e[1] == p2]
                                        if len(atomic_2_mutants)==0 : continue
                                        dict_presence[m][d][s][y][cn] = 0


                            error0 = intersectional_mutant[2]
                            error1 = dict_presence[m][d][s][x][cn]
                            error2 = dict_presence[m][d][s][y][cn]

                            if error0:
                                dict_hidden_error[m][d]["Number Errors"] += 1
                                if not error1 and not error2 :
                                    dict_hidden_error[m][d]["Hidden Errors"] += 1
                                    hidden_ones.append((intersectional_mutant, m, d , s, cn))
            dict_hidden_error[m][d]["Hidden Error rate"] = dict_hidden_error[m][d]["Hidden Errors"]/dict_hidden_error[m][d]["Number Errors"] if dict_hidden_error[m][d]["Number Errors"] != 0 else "/"


#Number errors
results_lines.append([])
results_lines.append([])
results_lines.append(["Number errors d/m"])
results_lines.append([""]+all_models)
for d in dts:
    line = [d]
    for m in all_models:
        if d not in dict_hidden_error[m] :
            line += ["/"]
        else:
            line += [dict_hidden_error[m][d]["Number Errors"]]
    results_lines.append(line)



#Hidden errors
results_lines.append([])
results_lines.append([])
results_lines.append(["Hidden errors d/m"])
results_lines.append([""]+all_models)
for d in dts:
    line = [d]
    for m in all_models:
        if d not in dict_hidden_error[m] :
            line += ["/"]
        else:
            line += [dict_hidden_error[m][d]["Hidden Errors"]]
    results_lines.append(line)


#Hidden rate
results_lines.append([])
results_lines.append([])
results_lines.append(["Hidden rates d/m"])
results_lines.append([""]+all_models)
dict_total_model_hidden = dict()
for m in all_models:
    dict_total_model_hidden[m] = dict()
    dict_total_model_hidden[m]["Hidden Errors"] = 0
    dict_total_model_hidden[m]["Number Errors"] = 0
for d in dts:
    line = [d]
    for m in all_models:
        if d not in dict_hidden_error[m] :
            line += ["/"]
        else:
            line += [dict_hidden_error[m][d]["Hidden Error rate"]]
            dict_total_model_hidden[m]["Hidden Errors"] += dict_hidden_error[m][d]["Hidden Errors"]
            dict_total_model_hidden[m]["Number Errors"] += dict_hidden_error[m][d]["Number Errors"]
    results_lines.append(line)
model_hidden_rates = [dict_total_model_hidden[m]["Hidden Errors"]/dict_total_model_hidden[m]["Number Errors"] if dict_total_model_hidden[m]["Number Errors"] != 0 else "/" for m in all_models]
results_lines.append(["Total"]+model_hidden_rates)

################### Write #############################
methods.writeCSV(results_path, results_lines)
print("done")
