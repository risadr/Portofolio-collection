import json
from collections import OrderedDict
 
###
# Create overall  dataset descriptor
###

data = OrderedDict()

#name of the dataset
data['name'] = 'switching task'

#The version of the BIDS standard that was used
data['BIDSVersion']='1.0.2'

#recommended fields:
#what license is this dataset distributed under? The use of license name abbrevi
data['license']=''

#List of individual who contributed to the creation / curation of the data set
data['Authors'] = ['Risa Dwi Ratnasari', 'Tzu-Yu Hsu']

#who should be acknowledge in helping to collect the data
data['Acknowledgements'] = 'Hsin-Yi'

#Instruction how researchers using this dataset should acknowledge the original
data['HowToAcknowledge'] = ''

#sources of funding (grant numbers) 
data['Funding'] = ['']

#a list of references to publication that contain information on the dataset, or links.
data['ReferencesAndLinks'] = ['','','']

#the Document Object Identifier of the dataset (not the corresponding paper).
data['DatasetDOI'] = ''


root_dir = '/Users/risadwiratnasari/Practice_arrange_data/'
project_label = "switching_task_three_participants"


dataset_json_folder = root_dir+project_label
dataset_json_name=dataset_json_folder+'/'+'dataset_description.json'

with open(dataset_json_name, 'w') as ff:
    json.dump(data, ff,sort_keys=False, indent=4)




# Add bidsignore file for MRS data and derivatives folder
#####
# f = open(dataset_json_folder+'/.bidsignore','w')
# f.write('derivatives/\n')
# f.close()

#####
# Create subject info file descriptor
#####

info = OrderedDict()

info['participant_id'] = {"Description":'ID number for participant'}

info['age'] = {"Description":"Participant's age",
                "Units":"Years"}

info['sex'] = {"Description": "Participant's sex",
                "Levels": {"m":"male","f":"female"}
                }

info['experiment_date'] = {"Description":"Day on which experiment was performed"}

# Write file
with open(dataset_json_folder+'/participants.json', 'w') as ff:
    json.dump(info, ff,sort_keys=False, indent=4)



#####
# Create task descriptors
#####

### switching
task = OrderedDict()

# The name of the task
task['TaskName'] = 'switching-task'

# Write file
with open(dataset_json_folder+'/task-switching.json', 'w') as ff:
    json.dump(task, ff,sort_keys=False, indent=4)

### Button press
task = OrderedDict()

# The Variety of the task
task['TaskVariety'] = 'preference','similarity'

# Write file
with open(dataset_json_folder+'/task-switching.json', 'w') as ff:
    json.dump(task, ff,sort_keys=False, indent=4)


