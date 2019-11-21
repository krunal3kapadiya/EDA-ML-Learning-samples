
# FIFA19 Explanatory Data Analysis


![](http://i.imgur.com/EzhxngF.jpg)

What will you will find in this Kernel
- Q 1. Average, maximum and minimum players count.
- Q 2. Age vs Potential
- Q 3. Average potential by age
- Q 4. Players joinee as per year
- Q 5. Players joinee as per month
- Q 6. Height and dribblling
- Q 7. FK Accuracy and Heading Accuracy
- Q 8. Lefty and Righty player
- Q 9. Valid contracts
- Q 10. Overall aggrassion


```python
# importing libraries
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
```


```python
# reading dataset
df_data = pd.read_csv('../input/data.csv')
df_data.head() #printing values in dataset
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>ID</th>
      <th>Name</th>
      <th>Age</th>
      <th>Photo</th>
      <th>Nationality</th>
      <th>Flag</th>
      <th>Overall</th>
      <th>Potential</th>
      <th>Club</th>
      <th>...</th>
      <th>Composure</th>
      <th>Marking</th>
      <th>StandingTackle</th>
      <th>SlidingTackle</th>
      <th>GKDiving</th>
      <th>GKHandling</th>
      <th>GKKicking</th>
      <th>GKPositioning</th>
      <th>GKReflexes</th>
      <th>Release Clause</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>158023</td>
      <td>L. Messi</td>
      <td>31</td>
      <td>https://cdn.sofifa.org/players/4/19/158023.png</td>
      <td>Argentina</td>
      <td>https://cdn.sofifa.org/flags/52.png</td>
      <td>94</td>
      <td>94</td>
      <td>FC Barcelona</td>
      <td>...</td>
      <td>96.0</td>
      <td>33.0</td>
      <td>28.0</td>
      <td>26.0</td>
      <td>6.0</td>
      <td>11.0</td>
      <td>15.0</td>
      <td>14.0</td>
      <td>8.0</td>
      <td>€226.5M</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>20801</td>
      <td>Cristiano Ronaldo</td>
      <td>33</td>
      <td>https://cdn.sofifa.org/players/4/19/20801.png</td>
      <td>Portugal</td>
      <td>https://cdn.sofifa.org/flags/38.png</td>
      <td>94</td>
      <td>94</td>
      <td>Juventus</td>
      <td>...</td>
      <td>95.0</td>
      <td>28.0</td>
      <td>31.0</td>
      <td>23.0</td>
      <td>7.0</td>
      <td>11.0</td>
      <td>15.0</td>
      <td>14.0</td>
      <td>11.0</td>
      <td>€127.1M</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>190871</td>
      <td>Neymar Jr</td>
      <td>26</td>
      <td>https://cdn.sofifa.org/players/4/19/190871.png</td>
      <td>Brazil</td>
      <td>https://cdn.sofifa.org/flags/54.png</td>
      <td>92</td>
      <td>93</td>
      <td>Paris Saint-Germain</td>
      <td>...</td>
      <td>94.0</td>
      <td>27.0</td>
      <td>24.0</td>
      <td>33.0</td>
      <td>9.0</td>
      <td>9.0</td>
      <td>15.0</td>
      <td>15.0</td>
      <td>11.0</td>
      <td>€228.1M</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>193080</td>
      <td>De Gea</td>
      <td>27</td>
      <td>https://cdn.sofifa.org/players/4/19/193080.png</td>
      <td>Spain</td>
      <td>https://cdn.sofifa.org/flags/45.png</td>
      <td>91</td>
      <td>93</td>
      <td>Manchester United</td>
      <td>...</td>
      <td>68.0</td>
      <td>15.0</td>
      <td>21.0</td>
      <td>13.0</td>
      <td>90.0</td>
      <td>85.0</td>
      <td>87.0</td>
      <td>88.0</td>
      <td>94.0</td>
      <td>€138.6M</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>192985</td>
      <td>K. De Bruyne</td>
      <td>27</td>
      <td>https://cdn.sofifa.org/players/4/19/192985.png</td>
      <td>Belgium</td>
      <td>https://cdn.sofifa.org/flags/7.png</td>
      <td>91</td>
      <td>92</td>
      <td>Manchester City</td>
      <td>...</td>
      <td>88.0</td>
      <td>68.0</td>
      <td>58.0</td>
      <td>51.0</td>
      <td>15.0</td>
      <td>13.0</td>
      <td>5.0</td>
      <td>10.0</td>
      <td>13.0</td>
      <td>€196.4M</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 89 columns</p>
</div>




```python
df_data.columns
```




    Index(['Unnamed: 0', 'ID', 'Name', 'Age', 'Photo', 'Nationality', 'Flag',
           'Overall', 'Potential', 'Club', 'Club Logo', 'Value', 'Wage', 'Special',
           'Preferred Foot', 'International Reputation', 'Weak Foot',
           'Skill Moves', 'Work Rate', 'Body Type', 'Real Face', 'Position',
           'Jersey Number', 'Joined', 'Loaned From', 'Contract Valid Until',
           'Height', 'Weight', 'LS', 'ST', 'RS', 'LW', 'LF', 'CF', 'RF', 'RW',
           'LAM', 'CAM', 'RAM', 'LM', 'LCM', 'CM', 'RCM', 'RM', 'LWB', 'LDM',
           'CDM', 'RDM', 'RWB', 'LB', 'LCB', 'CB', 'RCB', 'RB', 'Crossing',
           'Finishing', 'HeadingAccuracy', 'ShortPassing', 'Volleys', 'Dribbling',
           'Curve', 'FKAccuracy', 'LongPassing', 'BallControl', 'Acceleration',
           'SprintSpeed', 'Agility', 'Reactions', 'Balance', 'ShotPower',
           'Jumping', 'Stamina', 'Strength', 'LongShots', 'Aggression',
           'Interceptions', 'Positioning', 'Vision', 'Penalties', 'Composure',
           'Marking', 'StandingTackle', 'SlidingTackle', 'GKDiving', 'GKHandling',
           'GKKicking', 'GKPositioning', 'GKReflexes', 'Release Clause'],
          dtype='object')




```python
def display_graph(ax, title, xlabel, ylabel, legend):
    '''
    Graph theme will be same throught the kernel
    '''
    ax.set_xlabel(xlabel, fontsize=16)
    ax.set_ylabel(ylabel, fontsize=16)
    plt.title(title, fontsize=18, fontweight='bold')
    plt.legend(legend)
    plt.show()
```

##### Q 1. Average, maximum and minimum players count.


```python
print('In the age column there is total {} number of players and in dataset there is {} number of null in Age. Also in that data the mean (average age) is {}, maximum age is {}, and minimum age is {}, containing total {} numbers of countries. Now, lets display the other data related information based on Age.'.format(df_data['Age'].sum(), df_data['Age'].isna().sum(), format(df_data['Age'].mean(), '.2f'), df_data['Age'].max(), df_data['Age'].min(), len(df_data['Nationality'].unique())))
```

    In the age column there is total 457400 number of players and in dataset there is 0 number of null in Age. Also in that data the mean (average age) is 25.12, maximum age is 45, and minimum age is 16, containing total 164 numbers of countries. Now, lets display the other data related information based on Age.



```python
print('This dataset have {} players having age 16 and {} players who has age more than 42'.format(sum((df_data['Age'] == 16)),sum(df_data['Age'] >= 40)))
```

    This dataset have 42 players having age 16 and 22 players who has age more than 42



```python
ax = sns.distplot(df_data[['Age']])
display_graph(ax, 'Age', 'Age count', '', ['Age'])
```


![png](graphs/output_10_0.png)


###### Insights:
From above graph, we can see that average players count are between 21 to 27. There are sudden rise between players 15 to 21, it means that there are less number of players between this ages. But at the tail side, we can see that graph is slowly descreasing, so the players age are slightly decreasing reaching to the age 45


```python
ax = sns.distplot(df_data[['Potential']])
display_graph(ax, 'Potential count', 'Potential', '', ['Potential'])
```


![png](graphs/output_12_0.png)


##### Q 2. Age vs Potential


```python
ax = sns.scatterplot(x = 'Age', y='Potential', data=pd.DataFrame(df_data, columns=['Age', 'Potential']))
display_graph(ax, 'Age vs Potential graph', 'Age', 'Potential', ['Age'])
```


![png](graphs/output_14_0.png)


###### Insights
As we can see in this graph, it is clear that there data scattered in the plot above age 40 are very less, it also do not have higher potential. Calculating the average, those players who has age between 20 to 30, can have perform more better than any other ages of players.


```python
df_age = pd.DataFrame(df_data, columns=['Name', 'Age', 'Potential', 'Nationality'])
df_age.sort_values(by='Age').head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Name</th>
      <th>Age</th>
      <th>Potential</th>
      <th>Nationality</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>18206</th>
      <td>G. Nugent</td>
      <td>16</td>
      <td>66</td>
      <td>England</td>
    </tr>
    <tr>
      <th>17743</th>
      <td>J. Olstad</td>
      <td>16</td>
      <td>69</td>
      <td>Norway</td>
    </tr>
    <tr>
      <th>13293</th>
      <td>H. Massengo</td>
      <td>16</td>
      <td>75</td>
      <td>France</td>
    </tr>
    <tr>
      <th>16081</th>
      <td>J. Italiano</td>
      <td>16</td>
      <td>79</td>
      <td>Australia</td>
    </tr>
    <tr>
      <th>18166</th>
      <td>N. Ayéva</td>
      <td>16</td>
      <td>72</td>
      <td>Sweden</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_age.sort_values(by='Age').tail()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Name</th>
      <th>Age</th>
      <th>Potential</th>
      <th>Nationality</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>12192</th>
      <td>H. Sulaimani</td>
      <td>41</td>
      <td>63</td>
      <td>Saudi Arabia</td>
    </tr>
    <tr>
      <th>10545</th>
      <td>S. Narazaki</td>
      <td>42</td>
      <td>65</td>
      <td>Japan</td>
    </tr>
    <tr>
      <th>18183</th>
      <td>K. Pilkington</td>
      <td>44</td>
      <td>48</td>
      <td>England</td>
    </tr>
    <tr>
      <th>17726</th>
      <td>T. Warner</td>
      <td>44</td>
      <td>53</td>
      <td>Trinidad &amp; Tobago</td>
    </tr>
    <tr>
      <th>4741</th>
      <td>O. Pérez</td>
      <td>45</td>
      <td>71</td>
      <td>Mexico</td>
    </tr>
  </tbody>
</table>
</div>



###### Insights
This is age distribution, of younger and older player as per country wise. To check which country have the youngest and oldest players. So Maxico have the oldest player, named as O. Perez.

##### Q 3. Average potential by age


```python
df_age.groupby('Age', as_index=False).count().head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Name</th>
      <th>Potential</th>
      <th>Nationality</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>16</td>
      <td>42</td>
      <td>42</td>
      <td>42</td>
    </tr>
    <tr>
      <th>1</th>
      <td>17</td>
      <td>289</td>
      <td>289</td>
      <td>289</td>
    </tr>
    <tr>
      <th>2</th>
      <td>18</td>
      <td>732</td>
      <td>732</td>
      <td>732</td>
    </tr>
    <tr>
      <th>3</th>
      <td>19</td>
      <td>1024</td>
      <td>1024</td>
      <td>1024</td>
    </tr>
    <tr>
      <th>4</th>
      <td>20</td>
      <td>1240</td>
      <td>1240</td>
      <td>1240</td>
    </tr>
  </tbody>
</table>
</div>




```python
ax = df_age.groupby('Age').mean().plot.bar()
display_graph(ax, 'Average potential count', 'Age', 'Potential', ['Potential'])
```


![png](graphs/output_21_0.png)


##### Insights
Again there is plot of average age and potential graph. To plot it in bar chart, to check the overall potential score. We can see that those player have age 44 have less potential and those who have age 45 have higher potential. But again, We are considered mean value. This plot contains, overall performance based on age.

##### 4. Players Joinee as per year


```python
df_joined = df_data['Joined']
```


```python
df_joined.isna().sum()
```




    1553




```python
df_joined.dropna(inplace = True)
```


```python
df_joined = df_joined.apply(lambda x: datetime.strptime(x, '%b %d, %Y'))
```


```python
# get the list of years
df_year = df_joined.apply(lambda x: x.year)
```


```python
ax = df_year.value_counts().plot()
display_graph(ax, 'Players growth over the year', 'Years', 'Number of players', ['Players Growth'])
```


![png](graphs/output_29_0.png)


##### Insights
As we can see in the graph, based on data there are sudden rise of fifa player after the year 2014. Other things are self explanatory. Isn't it? :smile:

##### Q 5. Players joinee as per month


```python
df_month = df_joined.apply(lambda x: x.month)
```


```python
df_month.sort_values(ascending = True, inplace=True)
```


```python
ax = df_month.value_counts(sort = False).plot.bar()
display_graph(ax, 'Enrolling players as per month', 'Month', 'Number of players', ['Player/Month'])
```


![png](graphs/output_34_0.png)


##### Insights
So, coming to the year wise enrollment. In july month there are sudden rise enrollment of players, and in April,  November have the lowest value of forms to enroll the players.

##### Q 6. Height and dribblling


```python
df_height = pd.DataFrame(df_data, columns=['Height', 'Weight', 'Strength', 'Aggression', 'Stamina', 'Dribbling'])
```


```python
df_height.corr()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Strength</th>
      <th>Aggression</th>
      <th>Stamina</th>
      <th>Dribbling</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Strength</th>
      <td>1.000000</td>
      <td>0.474120</td>
      <td>0.262694</td>
      <td>-0.033550</td>
    </tr>
    <tr>
      <th>Aggression</th>
      <td>0.474120</td>
      <td>1.000000</td>
      <td>0.645687</td>
      <td>0.441075</td>
    </tr>
    <tr>
      <th>Stamina</th>
      <td>0.262694</td>
      <td>0.645687</td>
      <td>1.000000</td>
      <td>0.686511</td>
    </tr>
    <tr>
      <th>Dribbling</th>
      <td>-0.033550</td>
      <td>0.441075</td>
      <td>0.686511</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_height.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Strength</th>
      <th>Aggression</th>
      <th>Stamina</th>
      <th>Dribbling</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>18159.000000</td>
      <td>18159.000000</td>
      <td>18159.000000</td>
      <td>18159.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>65.311967</td>
      <td>55.868991</td>
      <td>63.219946</td>
      <td>55.371001</td>
    </tr>
    <tr>
      <th>std</th>
      <td>12.557000</td>
      <td>17.367967</td>
      <td>15.894741</td>
      <td>18.910371</td>
    </tr>
    <tr>
      <th>min</th>
      <td>17.000000</td>
      <td>11.000000</td>
      <td>12.000000</td>
      <td>4.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>58.000000</td>
      <td>44.000000</td>
      <td>56.000000</td>
      <td>49.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>67.000000</td>
      <td>59.000000</td>
      <td>66.000000</td>
      <td>61.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>74.000000</td>
      <td>69.000000</td>
      <td>74.000000</td>
      <td>68.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>97.000000</td>
      <td>95.000000</td>
      <td>96.000000</td>
      <td>97.000000</td>
    </tr>
  </tbody>
</table>
</div>



##### Q 7. FK Accuracy and Heading Accuracy


```python
accuracy = pd.DataFrame(df_data, columns=['HeadingAccuracy', 'FKAccuracy'])
```


```python
accuracy.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>HeadingAccuracy</th>
      <th>FKAccuracy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>70.0</td>
      <td>94.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>89.0</td>
      <td>76.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>62.0</td>
      <td>87.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>21.0</td>
      <td>19.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>55.0</td>
      <td>83.0</td>
    </tr>
  </tbody>
</table>
</div>



##### Q 8. Lefty and Righty player


```python
prefered_type = df_data['Preferred Foot'].value_counts()
prefered_type
```




    Right    13948
    Left      4211
    Name: Preferred Foot, dtype: int64




```python
sum(df_data['Preferred Foot'].isnull())
```




    48




```python
ax = prefered_type.plot.bar()
display_graph(ax, 'Righty/Lefty Players count', 'Preffered leg', 'Number of players', ['Right', 'Left'])
```


![png](graphs/output_46_0.png)


##### Insights
We have total 48 empty values in preferred foot other than that, you can see in graph that we have almost 14000 (precisely 13948) players who is righty and above 4000 (accuratly 4211) lefty players. Now let's plot this players as which country has more lefty/righty players

##### Q 9. Valid contracts


```python
df_data['Contract Valid Until'].value_counts().head(10)
```




    2019            4819
    2021            4360
    2020            4027
    2022            1477
    2023            1053
    Jun 30, 2019     931
    2018             886
    Dec 31, 2018     144
    May 31, 2019      60
    Jan 1, 2019       51
    Name: Contract Valid Until, dtype: int64




```python
df_contract = pd.DataFrame(df_data, columns=['Contract Valid Until'])
```


```python
df_contract.dropna(inplace = True)
```


```python
def get_only_year(dates):
    '''
    some of the date in this df contains 21 Jul, 2018 and some have only names
    so, getting only years value
    '''
    newDates = []
    for i, date in enumerate(dates):
        if(len(date)>4):
            date = date[-4:]
        newDates.append(date)
    return newDates
```


```python
df_contract_valid = get_only_year(df_contract['Contract Valid Until'])
```


```python
df_contract_valid = pd.Series(df_contract_valid)
```


```python
len(df_contract_valid.unique())
```




    9




```python
ax = df_contract_valid.value_counts().plot()
display_graph(ax, 'Contract valid until', 'Years', 
             'Players count', ['Contract'])
```


![png](graphs/output_56_0.png)


##### Insights
As per the data, above 5800 players' contract ending in the 2019. In 2021, it decreses to 4100/4200, then there is slightly drop to 4000 in year 2023 and massive drop of contract occurs at 2022 upto 1500.

##### Q 10. Overall aggrassion


```python
f = (df_data
         .loc[df_data['Position'].isin(['ST', 'GK'])]
         .loc[:, ['Value', 'Overall', 'Aggression', 'Position']]
    )
f = f[f["Overall"] >= 80]
f = f[f["Overall"] < 85]
f['Aggression'] = f['Aggression'].astype(float)
```


```python
ax = sns.boxplot(x="Overall", y="Aggression", hue='Position', data=f)
display_graph(ax, 'Overall Aggression', 'Overall', 'Aggression', ['ST, GK'])
```


![png](graphs/output_60_0.png)


### Conclusion
So, in this dataset, I have explained the rows related to age, potential, accuracy, contract and the preffered type of the player and many more. Major explaination is described in the insights of the charts. This data can be further explained with showing the information of the data as per the country and club wise.

--------------------------

~If you like this kernel please give star to it. Also, follow me on [Twitter](https://twitter.com/krunal3kapadiya) or [Medium](https://medium.com/@krunal3kapadiya) for more updates. You can also check my website https://krunal3kapadiya.app ~
