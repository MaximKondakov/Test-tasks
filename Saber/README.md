# Saber_test

The notebook performs data analysis tasks using SQL and merges JSON logs using Python. Key functionalities include calculating average time tasks are in an "Open" status, retrieving information about currently open tasks, and merging JSON log files.

## Contents
### SQL Queries:
1. Average Time in "Open" Status:
Calculate the average time tasks from different groups spend in the "Open" status.
Group tasks by the first character of the task key and convert time to hours.
2. Retrieve Currently Open Tasks:
Retrieve tasks that are currently open, excluding those with the status "Closed" or "Resolved".
Use a subquery to find the last known status and creation time for tasks based on a given date.
              
### Log Processing with Python:
1. Classes and Functions:
Define classes and functions to read, parse, and sort JSON logs based on timestamps.
2. Argument Parsing:
Use argparse to handle input arguments for file paths.
3. Merging JSON Logs:
Implement a function to merge and sort JSON log entries from multiple files.
Write the merged result to a specified output file.
