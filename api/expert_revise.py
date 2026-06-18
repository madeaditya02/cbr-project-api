import csv

def pending_cases():
  # read file need_revise_case.csv and get data with status pending and return it
  with open('need_revise_case.csv', 'r') as file:
    reader = csv.reader(file)
    data = list(reader)
    pending = []
    for row in data:
      if row[-1] == 'pending':
        pending.append(row)
    return pending
    