import os
import pathlib

INPUT_FILES_FOLDER = './RawData'
OUTPUT_FILES_FOLDER = '../InputData'

def create_output_folder():
    if(not os.path.isdir(OUTPUT_FILES_FOLDER)):
        pathlib.Path(OUTPUT_FILES_FOLDER).mkdir(parents=True, exist_ok=True)

def build_company_close_day(companies_close_data, company_close_day):
    comany_name = company_close_day[0]
    if not(comany_name in companies_close_data.keys()):
        companies_close_data[comany_name] = []

    company = companies_close_data[comany_name]
    company.append(company_close_day[1] + ','
        + company_close_day[2] + ','
        + company_close_day[3] + ','
        + company_close_day[4] + ','
        + company_close_day[5] + ','
        + company_close_day[6])

def close_data_to_csv(companies_close_data):
    for company_name, company in companies_close_data.items():
        company.sort(key = lambda line: line.split(',')[0])
        with open(OUTPUT_FILES_FOLDER + '/' + company_name + '.csv', 'w') as file:
            for line in company:
                file.write("%s" % line)

create_output_folder()
companies_close_data = {}

files = os.listdir(INPUT_FILES_FOLDER)
for file_name in files:
    file = open(INPUT_FILES_FOLDER + '/' + file_name, "r")
    for line in file.readlines():
        company_close_day = line.split(',')
        if(len(company_close_day) == 7):
            build_company_close_day(companies_close_data, company_close_day)

    file.close()

close_data_to_csv(companies_close_data)