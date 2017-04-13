import json
import numpy as np
from TFFunctions import *
from numpy import genfromtxt


file_str = "/mnt/tt2botdata/Export_switchRuleWirelessPackage004_58eb9cfbe4b01ec9bc406f48.json"
file_cvs_str = "/mnt/tt2botdata/SwitchRule_58eb9cfbe4b01ec9bc406f48.csv"
pickle_save_str = "/mnt/tt2botdata/data_pickle"
file_on = open(file_str)

jrules = json.load(file_on)
rules_list = jrules["SwitchRuleWirelessPackage004Entity"]
print rules_list[0]
for field in rules_list[0]:
    print field
print "product line: " + rules_list[0]["productLine"]
print "gracePeriod: " + str( rules_list[0]["gracePeriod"])

print "calculating unique values"
product_line_values = []
source_bundle_values = []
target_bundle_values = []
segment_values = []
channel_values = []
use_case_values = []

#calculate distict values
for rule in rules_list:
    if rule["productLine"] not in product_line_values:
        product_line_values.append(rule["productLine"])
    if rule["targetBundleName"] not in target_bundle_values:
        target_bundle_values.append(rule["targetBundleName"])
    if rule["originalBundleName"] not in source_bundle_values:
        source_bundle_values.append(rule["originalBundleName"])
    if rule["orderChannel"] not in channel_values:
        channel_values.append(rule["orderChannel"])
    if rule["useCase"] not in use_case_values:
        use_case_values.append(rule["useCase"])
    if rule["customerSegment"] not in segment_values:
        segment_values.append(rule["customerSegment"])


num_features = 5 # key fields or raw inputs to the engine
dataset = np.ndarray((len(rules_list), num_features), dtype=np.float32)
labels = np.ndarray((len(rules_list)), dtype=np.int32)


print product_line_values
print source_bundle_values
print target_bundle_values
print use_case_values
print channel_values
print segment_values

print segment_values.index("RES")
print len(segment_values)


#transform to feature to numbers
rule_counter = 0
for rule in rules_list:
    dataset[rule_counter, 0] = ((source_bundle_values.index(rule["originalBundleName"]) + 1) / float(len(source_bundle_values)))
    dataset[rule_counter, 1] = ((target_bundle_values.index(rule["targetBundleName"]) + 1) / float(len(target_bundle_values)))
    dataset[rule_counter, 2] = ((channel_values.index(rule["orderChannel"]) + 1) / float(len(channel_values)))
    dataset[rule_counter, 3] = ((use_case_values.index(rule["useCase"]) + 1) / float(len(use_case_values)))
    dataset[rule_counter, 4] = ((segment_values.index(rule["customerSegment"]) + 1) / float(len(segment_values)))
    labels[rule_counter] = product_line_values.index(rule["productLine"])
    rule_counter += 1




print dataset
print labels
print dataset.shape

dataset, labels = randomize(dataset, labels)
# print("dataset shape:",dataset.shape,"dataset data",dataset)
# print("labels shape:",labels.shape,"labels data",labels)
# separate for train/test/validation
size_train = int(len(rules_list) * 0.8)
size_test_val = int(len(rules_list) * 0.1)
end_train_index = size_train
start_valid_index = end_train_index + 1
end_valid_index = start_valid_index + size_test_val
start_test_index = end_valid_index + 1
end_test_index = start_test_index + size_test_val

#print("end_train_index: ", end_train_index, " end_valid_index: ", end_valid_index, " end_test_index: ",
#      end_test_index)

train_dataset = dataset[0:end_train_index, :]
train_labels = labels[0:end_train_index]
valid_dataset = dataset[start_valid_index:end_valid_index, :]
valid_labels = labels[start_valid_index:end_valid_index]
test_dataset = dataset[start_test_index:end_test_index, :]
test_labels = labels[start_test_index:end_test_index]

save = {
    'train_dataset': train_dataset,
    'train_labels': train_labels,
    'valid_dataset': valid_dataset,
    'valid_labels': valid_labels,
    'test_dataset': test_dataset,
    'test_labels': test_labels,
    'product_line_classes': len(product_line_values),
    'num_examples': len(rules_list)
}

save_pickle(pickle_save_str, save)




print "done"