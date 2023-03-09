# Predicting Risk of Admission to the Emergency Department using MIMIC-IV

This repo describes how to generate a dataset for the purposes of training risk prediction models that assess the risk of inpatient admission in an emergency department. 

## MIMIC 4 Dataset

[MIMIC-IV (Medical Information Mart for Intensive Care)](https://mimic.mit.edu/docs/iv/) is a large, freely-available database comprising deidentified health-related data from patients who were admitted to the critical care units of the Beth Israel Deaconess Medical Center. 
Information related to the description and structures tables/features, up-to-date revisions, and a tutorial on how to get started using the data are available from the main site.

We used 4 files from MIMIC 4, which include admission and patients file under the `Core` directory, and triage and edstay file under Ed directory. 
You can either access the data using BigQuery from google and read them from [Google Healthcare datathon repository](https://github.com/GoogleCloudPlatform/healthcare), or you can direcly access and download the files from [Physionet](https://physionet.org/content/mimiciv/1.0/) after you complete the necessary setup including registering an accont, signing the data use agreement, and finishing the required training.

# Data Preprocessing

## Dependencies

- python 
- scikit-learn 
- pandas

To generate a single dataset from the repository, use the script `clean_mimic.py`. 
There are several options that can be passed to the script:

```
python clean_mimic.py -h

usage: clean_mimic.py [-mimic_path MIMIC_PATH] [-Admission_File ADMISSION_FILE] [-Edstay_File EDSTAY_FILE] [-Triage_File TRIAGE_FILE] [-Patient_File PATIENT_FILE] [-h] [-p PATH]

Input the file location for the four files from MIMIC IV.

options:
  -mimic_path MIMIC_PATH
                        Path for admission file
  -Admission_File ADMISSION_FILE
                        Path for admission file
  -Edstay_File EDSTAY_FILE
                        Path for edstay file
  -Triage_File TRIAGE_FILE
                        Path for Triage File
  -Patient_File PATIENT_FILE
                        Path for Patient file
  -h, --help            Show this help message and exit.
  -p PATH               Path of Saved final file
```

It is necessary that the tables for admissions, edstays, triage, and patients are available from MIMIC-IV. 
If those files are in the same directory structure as they are in MIMIC-IV, one can just pass `-mimic_path` and the script should find the correct file paths. 
Once the input are provided to our clean_mimic.py script, it will process the file and save them with the filename passed with the flag `-p`. 

The script conducts the following preprocessing:

## Step 1 : data merging

We first join triage and edstay table on stay_id; then join the result with admission table on subject_id and finally join result with patient table to get gender and age info.

## Step 2 : Dropping Unnecessary Columns

We drop duplicates on `stay_id` (keeping first entry) then drop unnecessary columns for our modelling (e.g. `deathtime`)
We then remove outliers based on some pre-determined criteria (for example, the temperature should be between 95 and 105)
Finally we remove patients who are admitted (explained in the next section) with admission_types with 'OBSERVATION' in the name.

## Step 3 : Creating New Columns

We create 3 new columns: previous number of admission, previous number of visits, and our label `y` indicating whether one is admitted or not.
For the label `y` indicating whether or not the patient is admitted, we just simply defined as whether or not the column 'hadm_id' is na(then 0) or not(then 1).

For previous number of admission, it's just the number of admission for a given `subject_id` prior to the current visit. Simiarly, previous number of visits is just  the number of visits for a given `subject_id` prior to the current visit. 
Note that the number of visits should always be greater than or equal to number of admission, as someone who makes visits does not necessaily get admitted. 
We manually create these two labels since they show up in the related literature as relevant features.

## Step 4 : Transforming Data

We transform the text variable `chiefcomplaint` using bag of words. 
Specifically, we one-hot encoded all of the vocabulary(using top 100 only), and treated the rest as the infrequent symptoms.

Also note that to deal with `chiefcomplaint`, we used the latest feature of sklean's one-hot encoding to encode infrequent features, which necessitates a recent version of scikit-learn. 

We also tried one-hot encoding other categorical variables including admission_type,admission_location,language,insuance,martial status, and ethnicity.

We convert continuous age variable into 5 year bins.


# Resulting Dataset

The cleaned dataset contains 173,561 ED visits over the span of 2011-2019. 
Below we break down the admission rates by demographic groups. 

## Overall Outcomes by Demographic

|                  |                               | Admit        | Discharge    | P-Value (adjusted)   |
|------------------|-------------------------------|--------------|--------------|----------------------|
| n                |                               | 53589        | 119972       |                      |
| Ethnicity, n (%) | American Indian/Alaska Native | 152 (35.6)   | 275 (64.4)   | <0.001               |
|                  | Asian                         | 2075 (34.7)  | 3904 (65.3)  |                      |
|                  | Black/African American        | 5727 (13.7)  | 36217 (86.3) |                      |
|                  | Hispanic/Latino               | 2231 (13.9)  | 13826 (86.1) |                      |
|                  | Other                         | 2711 (30.1)  | 6301 (69.9)  |                      |
|                  | Unknown/Unable to Obtain      | 3595 (79.3)  | 938 (20.7)   |                      |
|                  | White                         | 37098 (38.8) | 58511 (61.2) |                      |
| Gender, n (%)    | F                             | 26200 (26.4) | 72893 (73.6) | <0.001               |
|                  | M                             | 27389 (36.8) | 47079 (63.2) |                      |

## Admission prevalence (Admissions/Total (%)), stratified by the intersection of ethnoracial group and gender

| Ethnoracial Group             |   Male            | Female            | Overall            |
|-------------------------------|-------------------|-------------------|--------------------|
| American Indian/Alaska Native | 70/257 (27%)      | 82/170 (48%)      | 152/427 (36%)      |
| Asian                         | 1043/3595 (29%)   | 1032/2384 (43%)   | 2075/5979 (35%)    |
| Black/African American        | 3124/27486 (11%)  | 2603/14458 (18%)  | 5727/41944 (14%)   |
| Hispanic/Latino               | 1063/10262 (10%)  | 1168/5795 (20%)   | 2231/16057 (14%)   |
| Other                         | 1232/5163 (24%)   | 1479/3849 (38%)   | 2711/9012 (30%)    |
| Unknown/Unable to Obtain      | 1521/2156 (71%)   | 2074/2377 (87%)   | 3595/4533 (79%)    |
| White                         | 18147/50174 (36%) | 18951/45435 (42%) | 37098/95609 (39%)  |
| Overall                       | 26200/99093 (26%) | 27389/74468 (37%) | 53589/173561 (31%) |
