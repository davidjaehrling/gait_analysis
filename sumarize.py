import pandas as pd 

res = pd.read_csv('JntKinematics-GaitID.csv')

# delete all rows not containing the string 'Mid' in the column 'trialname'
res = res[res['trialname'].str.contains('Mid')]


columns = [["trialname", "Trial"],
           ["Right_AnkleDorsiFlexion_td", "TD_Ankledorsiflexion"],
           ["Right_AnkleDorsiFlexion_to", "TO_Ankledorsiflexion"],
           ["Right_KneeFlexion_td", "TD_Kneeflexion"],
           ["Right_KneeFlexion_to", "TO_Kneeflexion"],
           ["Right_KneeAdduction_td", "TD_Kneeadduction"],
           ["Right_KneeAdduction_to", "TO_Kneeadduction"],
           ["Right_HipFlexion_td", "TD_Hipflexion"],
           ["Right_HipFlexion_to", "TO_Hipflexion"]]

# create a list containing only the first elements of each list entry
first_elements = [col[0] for col in columns]
res = res[first_elements]

# rename the columns based on the second element of each list entry
res.columns = [col[1] for col in columns]

# delete all characters of "Trial" except the first 3
res['Trial'] = res['Trial'].str[:3]

# average the values of rows with the same 'Trial' value
res = res.groupby('Trial').mean()
print(res)

# save the result to a new csv file
res.to_csv('results_mocap_summarized.csv')

