import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as sts
from bioinfokit.analys import stat

def medicineData():
    #load data
    df = pd.read_csv("D:\\pythonProject\\DispersionAnalysis\\Medicine_Data.csv")
    #print(df)

    #delete columns
    df.drop(["Выполнено", "группа-НВ(граница 130: 1 -более, 2-менее для мужчин, 120 - для жщин)", "ИЛ6 (граница 40: 1 -менее 40, 2-40 и более)",
             "группа-ИЛ6(разибить по 10, 100, 1000)", "Волна (2 - декабрь и середина января, 3 - май по сегодня)"], axis=1, inplace=True)
    #print(df)

    #get sample

    df = df.sort_values(by='Возраст')

    group1 = df[df['Возраст'] < 25]
    group2 = df[(df['Возраст'] >= 25) & (df['Возраст'] < 45)]
    group3 = df[(df['Возраст'] >= 45) & (df['Возраст'] < 65)]
    group4 = df[df['Возраст'] >= 65]

    group1.rename(columns={'Возраст': '< 30 лет'}, inplace=True)
    group2.rename(columns={'Возраст': '30 <= лет < 60'}, inplace=True)
    group3.rename(columns={'Возраст': '60 <= лет < 75'}, inplace=True)
    group4.rename(columns={'Возраст': '>= 75 лет'}, inplace=True)

    #print(group1['Результат-Гемоглобин (HGB)'], group2['Результат-Гемоглобин (HGB)'],
     #                               group3['Результат-Гемоглобин (HGB)'], group4['Результат-Гемоглобин (HGB)'])

    #univariate dispersion analysis
    (fvalue, pvalue) = sts.f_oneway(group1['Результат-Гемоглобин (HGB)'], group2['Результат-Гемоглобин (HGB)'],
                                    group3['Результат-Гемоглобин (HGB)'], group4['Результат-Гемоглобин (HGB)'])
    print(fvalue, pvalue)

    df = pd.DataFrame(list(zip(group1['Результат-Гемоглобин (HGB)'], group2['Результат-Гемоглобин (HGB)'],
                               group3['Результат-Гемоглобин (HGB)'], group4['Результат-Гемоглобин (HGB)'])),
                      columns=['младше 25', 'младше 45', 'младше 65', 'старше 65'])

    df_melt = pd.melt(df.reset_index(), id_vars=['index'], value_vars=['младше 25', 'младше 45', 'младше 65', 'старше 65'])
    print(df_melt.head())

    ax = sns.boxplot(data=df_melt, x="value", y="variable")
    ax.set_xlabel('результат гемоглобина')
    plt.show()

    df = pd.DataFrame(list(zip(group1['Результат-Гемоглобин (HGB)'], group2['Результат-Гемоглобин (HGB)'],
                               group3['Результат-Гемоглобин (HGB)'], group4['Результат-Гемоглобин (HGB)'])),
                      columns=['A', 'B', 'C', 'D'])
    df_melt = pd.melt(df.reset_index(), id_vars=['index'], value_vars=['A', 'B', 'C', 'D'])

    # replace column names
    df_melt.columns = ['index', 'treatments', 'value']
    res = stat()
    res.tukey_hsd(df=df_melt, res_var='value', xfac_var='treatments', anova_model='value ~ C(treatments)')
    print(res.tukey_summary)

def bfiData():
    df = pd.read_csv("bfi.csv")
    #print(df.columns)

    df = df.iloc[:, 1:26]
    #print(df.columns)

    import numpy as np
    i = 0
    for j in range(len(df.columns)):
        w, pvalue = sts.shapiro(df.iloc[:, i])
        #print("for ", str(df.columns[i]), " dispersion = ", np.var(df.iloc[:, i]))
        if pvalue < 0.001:
            del df[str(df.columns[i])]
            i -= 1
        i += 1
    #print(df.columns)

    example = df.iloc[:, 17:20]
    #print(example.columns)

    w, pvalue = sts.shapiro(example)
    print(w, pvalue)

    w, pvalue = sts.bartlett(example['N3'], example['N4'], example['N5'])
    print(w, pvalue)

    from statsmodels.graphics.factorplots import interaction_plot
    interaction_plot(x=example['N4'], trace=example['N3'], response=example['N5'])
    plt.show()

    res = stat()
    res.anova_stat(df=example, res_var='N5', anova_model='N5~C(N3)+C(N4)+C(N3):C(N4)')
    print(res.anova_summary)

    res.tukey_hsd(df=example, res_var='N5', xfac_var='N3', anova_model='N5~C(N3)+C(N4)+C(N3):C(N4)')
    print(res.tukey_summary)

bfiData()
