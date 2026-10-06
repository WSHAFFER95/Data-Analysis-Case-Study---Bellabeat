import numpy as np
import pandas as pd
import scipy.stats as scp
import scikit_posthocs as ph
import matplotlib.pyplot as plt

dailylsdf = pd.read_csv("Datasets/hourly_fitbit_sema_df_unprocessed.csv",low_memory=False)
breq2df = pd.read_csv("Datasets/breq.csv",low_memory=False)


# Per https://www.bls.gov/opub/ted/2023/time-spent-in-leisure-and-sports-activities-2022.htm: average exercise for
# women per day in the US is approximately ~0.26 hours per day.

# Filter on female users only.
fmask = (dailylsdf['gender']=='FEMALE')
# Filter exercise for activity logged OR >=180 cal burned OR >=2000 steps walked.
# (dailylsdf['activityType']!=None)| (first mask removed for testing)
exmask = (dailylsdf['activityType'].notnull())|(dailylsdf['calories']>=180)|(dailylsdf['distance']>=2000)
# Filter dataframe on BOTH masks.
dailylsdf_fil = dailylsdf[fmask&exmask]
subjects = dailylsdf_fil['id'].unique()
# Masking BREQ2 results to ensure only female participants.
breq2_fil = breq2df[breq2df['user_id'].isin(subjects)]
breq_subjects = breq2_fil['user_id'].unique()
print(breq_subjects)
# Note that we have fewer subjects who completed the survey.

# Evaluating time-of-day differences:
observations = []
for subject in subjects:
    dailylsdf_fil_id = dailylsdf_fil[dailylsdf_fil['id']==subject]
    exercise_times = dailylsdf_fil_id['hour'].to_numpy()
    time_buckets = [0,0,0,0,0,0]
    for time in exercise_times:
        if time <= 3.0:
            time_buckets[0] += 1
            continue
        elif time > 3.0 and time <= 7.0:
            time_buckets[1] += 1
            continue
        elif time > 7.0 and time <= 11.0:
            time_buckets[2] += 1
            continue
        elif time > 11.0 and time <= 15.0:
            time_buckets[3] += 1
            continue
        elif time > 15.0 and time <= 19.0:
            time_buckets[4] += 1
            continue
        else:
            time_buckets[5] += 1
    observations.append(time_buckets)
# Generating value for average exercise over the length of the study
val = 0
trimester_val = []
for subject in observations:
    val = sum(subject)/124
    # sum exercises and divide by # days in dataset; assuming 31 days per month (pessimistic estimate)
    trimester_val.append(val)
print(scp.tmean(trimester_val))
# 0.8381123058542413
print(scp.bayes_mvs(trimester_val, alpha=0.99))
# (Mean(statistic=np.float64(0.8381123058542413), minmax=(np.float64(0.4353292015149004), np.float64(1.2408954101935823))), 
# Variance(statistic=np.float64(0.6145829318739482), minmax=(np.float64(0.30544680691965415), np.float64(1.3216556089414258))), 
# Std_dev(statistic=np.float64(0.7758319362724211), minmax=(np.float64(0.5526724227964104), np.float64(1.1496328148332517))))
# Rough estimate from 2022 EU data (assuming typical exercise length is 1 hour) is 0.139; with these intervals, we can determine that
# these EU smartwatch users exercise almost double normal citizens.
# Calculation: (5*.05+3.5*.11+1.5*.19+0.5*.05+0.25*.11+0*0.49)/7 gives daily avg amt of exercise - assuming typical exercise length is
# approx. 1 hour, we get the value 0.139. Source: Sport_and_Physical_Activity_EBS525_volume_B - EU 2022.
bucketed_obs = [[],[],[],[],[],[]]
for individual in observations:
    for i in range(6):
        bucketed_obs[i].append(individual[i])

# Generate comparison graphs.
fig,axs = plt.subplots(1,6,sharex=True,sharey=True,tight_layout=True)
axs[0].hist(bucketed_obs[0],color='xkcd:navy blue',align='left')
axs[0].set_title('12-3AM')
axs[1].hist(bucketed_obs[1],color='xkcd:grey blue',align='left')
axs[1].set_title('4-7AM')
axs[2].hist(bucketed_obs[2],color='xkcd:gold',align='left')
axs[2].set_title('8-11AM')
axs[3].hist(bucketed_obs[3],color='xkcd:goldenrod',align='left')
axs[3].set_title('12-3PM')
axs[4].hist(bucketed_obs[4],color='xkcd:orange',align='left')
axs[4].set_title('4-7PM')
axs[5].hist(bucketed_obs[5],color='xkcd:navy',align='left')
axs[5].set_title('8-11PM')
plt.suptitle('Exercise Periods by Frequency')
plt.show()
# Figure above justifies using Kruskal-Willis and/or 
# Friedman Chi-Squared to assess difference of means - clearly we have many similarly shaped
# distributions, right-skewed. Moreover, independence is a reasonable assumption to make, given that we are
# confident in our data source.
fm_result = scp.friedmanchisquare(bucketed_obs[0],bucketed_obs[1],bucketed_obs[2],bucketed_obs[3],
                        bucketed_obs[4],bucketed_obs[5])
print(fm_result)
# FriedmanchisquareResult(statistic=np.float64(83.38043478260866), pvalue=np.float64(1.6448021687304858e-16))
npy_data = np.array(bucketed_obs)
# We then perform Nemenyi to evaluate the likelihood of difference of means.
nem_result = ph.posthoc_nemenyi_friedman(npy_data.T)
print(nem_result)
#               0             1         2             3             4         5
# 0  1.000000e+00  9.735090e-01  0.000067  7.117999e-08  5.237144e-12  0.000894
# 1  9.735090e-01  1.000000e+00  0.001864  5.271471e-06  1.125441e-09  0.015316
# 2  6.731709e-05  1.864075e-03  1.000000  7.999203e-01  7.691205e-02  0.992242
# 3  7.117999e-08  5.271471e-06  0.799920  1.000000e+00  7.158524e-01  0.430373
# 4  5.237144e-12  1.125441e-09  0.076912  7.158524e-01  1.000000e+00  0.013580
# 5  8.939028e-04  1.531630e-02  0.992242  4.303730e-01  1.357986e-02  1.000000
# We note in the above that 0 and 1 have differing means from almost every other time period.
# Moreover, 2 and 4 ALMOST differ, so when checking medians via Wilcoxon, we should ensure to examine
# 2 and 4. We first perform Kruskal to ensure that we have a difference to evaluate.
kw_result = scp.kruskal(bucketed_obs[0],bucketed_obs[1],bucketed_obs[2],bucketed_obs[3],
                        bucketed_obs[4],bucketed_obs[5])
print(kw_result)
# KruskalResult(statistic=np.float64(72.54105843348357), pvalue=np.float64(3.0299653901537866e-14))
# Indicates that at least ONE true median differs from the others. Next we can proceed with pairwise comparisons
# between times of day, via Wilcoxon.
print(scp.wilcoxon(bucketed_obs[2],bucketed_obs[4],alternative='less'))
# Via Wilcoxon, 8-11AM is stochastically dominated by 4-7PM.
print(scp.wilcoxon(bucketed_obs[4],bucketed_obs[5],alternative='greater'))
# Via Wilcoxon, 8-11PM is stochastically dominated by 4-7PM.
print(scp.wilcoxon(bucketed_obs[3],bucketed_obs[4],alternative='less'))
# Via Wilcoxon, 12-3PM is stochastically dominated by 4-7PM.
print(scp.wilcoxon(bucketed_obs[2],bucketed_obs[3],alternative='less'))
# Via Wilcoxon, 8-11AM is stochastically dominated by 12-3PM.
print(scp.wilcoxon(bucketed_obs[3],bucketed_obs[5],alternative='greater'))
# Via Wilcoxon, we fail to reject H0. (Note that it is VERY close though.)
print(scp.wilcoxon(bucketed_obs[0],bucketed_obs[4],alternative='less'))
# Via Wilcoxon, 12-3AM is stochastically dominated by 4-7PM.
print(scp.wilcoxon(bucketed_obs[1],bucketed_obs[4],alternative='less'))
# Via Wilcoxon, 4-7AM is stochastically dominated by 4-7PM.
# As 4-7PM stochastically dominates all other time periods, we can conclude that this
# is the most likely time period for female users to exercise, and that we should focus marketing efforts on
# this time period.

# We evaluate BREQ2 results next:
# Five types of exercise regulation as detailed by the BREQ2 survey information (http://exercise-motivation.bangor.ac.uk/breq/brqscore.php):
# Amotivation, external regulation, introjected regulation, identified regulation, and intrinsic regulation.
# We are seeking to evaluate which of these dominates among female users of smartwatches.

breq2_fil_unique = breq2_fil.drop_duplicates(subset=['user_id'],keep='first')
reg_types = breq2_fil_unique['breq_self_determination'].to_numpy()
bucketed_regulation = [0,0,0,0,0]
for reg in reg_types:
    if reg == 'amotivation':
        bucketed_regulation[0] += 1
        continue
    elif reg == 'external_regulation':
        bucketed_regulation[1] += 1
        continue
    elif reg == 'introjected_regulation':
        bucketed_regulation[2] += 1
        continue
    elif reg == 'identified_regulation':
        bucketed_regulation[3] += 1
        continue
    else:
        bucketed_regulation[4] += 1

print(bucketed_regulation)

# The vast majority of users who completed the survey have introjected, identified, or intrinsic regulation. 
# As these three regulation types have very different dependencies on alerts, it's recommended to ask users whether 
# they would like reminders or not on startup, so as not to annoy them with unnecessary alerts or fail to give them alerts at all when 
# they want them.
