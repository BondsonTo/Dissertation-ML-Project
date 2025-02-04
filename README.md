# ESG Score to Explain Company Financial Performance
This is my Master degree data analysis dissertation project.  Models built with various machine learning alglorithms trying to validate if companies ESG efforts would contribute to financial growth.

>**Technical Expertise: `Python`, `Power BI`, `Power Query`, `Google Colab`**

>**Skills: `Data Analysis`, `Machine Learning`, `Data Visualization`**

## Background

Environmental, Social and Governance (ESG) became increasingly important for businesses and investors due to the growing awareness. Despite of inter-governmental efforts to promote ESG, Investors are also paying more attention to ESG performance as their investment criteria. Some viewed compliance and reporting requirement associated with ESG effort would adversely affect the company financials. This project aimed at investigating if ESG efforts and transformation really paid off at company level.

## Abstract

This project investigates the relationship between **Environmental, Social, and Governance (ESG) scores** and C**orporate Financial Performance (CFP)** using machine learning models. With a dataset of 3,422 companies across 26 countries, the research deploys four machine learning techniques, including **Linear Regression, Random Forest, Support Vector Regression, and Artificial Neural Networks**, to predict Return on **Equity (ROE)** based on ESG scores. The findings indicate a notable correlation between high ESG scores and improved financial performance. ESG score with factor significance to explain company ROE comparable to company net sales. Such correlation is highly industry-specific with complex relationship remain unresolved. The findings support companies putting higher emphasis in ESG efforts and transformation.

## Quick glance of the dataset
### Dashboard Overview
The dataset for this project is retrieved from Refinitiv database as of 31 May 2024. Below **`Power BI`** dashboard summarized and visualised the key metrics after data preprocessing

![Power BI Dashboard](https://github.com/user-attachments/assets/6ee913f9-926c-4a6a-9911-ef0d9a9ca002)

>-	After data cleaning, **3513 companies** remained across 74 industries in final dataset. The industry classification followed the GICS as aligned with the Refinitiv database. [^1]
>-	**Return on Equity (ROE %)** is the financial measures as proxy of company financial performance. The average ROE% as per latest financial year [^2] in companies from final dataset is 11.28%.
>-	**ESG score** as proxy for company ESG performance measures. The score in this data analysis project referred to the score given by Refinitiv ranged from 0 to 100. There is a composite combined score (ESG combined score), which made of 3 individual pillar score
(Environmental, Social and Governance). In the final dataset, ESG combined score averaged 50.52. Individual pillar score of environmental, Social and Governance are 46.92, 56.23 and 55.35 respectively.
>-	No observable patterns in between **industry average ESG score and industry ROE in the scattered plot**. Hotel and REITS (63.35), Containers & Packaging (61.42), Chemicals (61.24) and Office REITS (60.93) are the top ESG performing industries (Green dots in scattered plot) measured with average ESG combined score. On the contrary, Mortgage Real Estate Investment Trusts (32.14), Entertainment (35.28) and Biotechnology (38.45) are the low ESG performing industries (Red dots in the scattered plot).
>![GICS Industry ROE vs ESG Score](https://github.com/user-attachments/assets/81da9166-f20b-419c-80a0-6d1e0fe2cdff)
>-	Companies in the final dataset scattered across **26 countries in America, Europe and Asia**, as shown in the dots in Geographic Distribution in the above dashboard.

### Data Preprocessing and Cleaning
The final dataset went through a 2-step process for data preprocessing and cleaning [^3]. It is to ensure robust and valid prediction to be built with machine learning algorithms.

(1)	Merging Refinitiv ROE, ESG score and financial data with macro-economic data from World Bank Open Data, using **`Power Query`** editor in **Excel**

a.	Country of Exchange is the key to merge country-wise GDP% growth, Inflation % and Unemployment % to each company in          final dataset. Those macro-economic data are included in subsequent prediction model building.
    
(2)	Drop and fill missing values and remove outliers in ROE using **`Python`** operated in **`Google Colab`** environment

a.	Companies in initial dataset with missing ROE or ESG Combined score was dropped
    
b.	Missing values in financial data were filled up with the average of data within the same GICS industry.
    
c.	Companies with extreme ROE% values, greater than 100% or less than 100% were dropped.


[^1]: Global Industry Classificaion Standard (GICS) is an industry analysis framework that helps investors understand the key business activities for companies around the world. The classification follows hierarchy starting from 11 sectors to 25 industry groups and then sub divided into 74 industries.
[^2]: ROE% is calculated from annual financial performance in financial year of the company. Companies would take various year end dates in financial announcement. To calculate this ROE % average, each ROE % correspond to latest available yearly data as in Refinitiv database. The year end date ranged from 31 Dec 2022 to 30 Apr 2024.
[^3]: Python codes for data preprocessing and cleaning can refer to the .py file in depository

## Deep dive into the dataset (Exploratory Data Analysis)

> [!NOTE]
> This session included substantial mathematical and statistical contents.  

### ROE%
![frequency_distribution_D (2)](https://github.com/user-attachments/assets/5fb02b38-db06-4561-9a07-b0d94f5403b8)

>ROE % followed an approximately bell shape where most observations clustered around mean, median and mode. Few extreme ROE% appeared on both positive and negative end of the graph with pronounced tail extensions. The graph appeared to be right-skewed with higher frequency of positive ROE % value.

![Box_Plot_ROE_by_industry (4)](https://github.com/user-attachments/assets/005f0d68-70a8-4d37-a36f-54b86c444bed)

>Segmented the data by GICS industries on ROE % in box plot gave another dimension for understanding. Notable ROE% outliers appeared in almost each of the dataset. Higher proportion of industries have averaged positive ROE%, except for biotechnology companies with whole IQR way below zero.

### Predictors Variables
The predictors are classified into 4 major groups: ESG metrics, financial metrics, macroeconomic metrics and industry classification[^4].

|**ESG metrics**|**Financial metrics**|**Macroeconomic metrics**|Industry classification|
|---------------|---------------------|-------------------------|---|
|ESG Combined Score<sub>[0][1][2]</sub>|Market Beta (5 Years monthly)|GDP Growth %<sub>[0]</sub>|GICS Industry[^1]|
|ESG Score<sub>[0][1][2]</sub>|Capital Expenditure<sub>[0][1][2]</sub>|Inflation %<sub>[0]</sub>|
|Environmental Score<sub>[0][1][2]</sub>|Current Ratio<sub>[0]</sub>|Unemployment %<sub>[0]</sub>|
|Social Score<sub>[0][1][2]</sub>|EBITDA <sub>[0][1][2]</sub>||
|Governmance Score<sub>[0][1][2]</sub>|Intangibles<sub>[0]</sub>||
||Market Capitalization||
||Net Sales<sub>[0]</sub>||
||PE Ratio<sub>[0][1][2]</sub>||
||Profit Margin %<sub>[0][1][2]</sub>||

Full list and explanation of each predictor variables can refer to [Appendix 1 Full lists of model response and predictors.pdf](https://github.com/user-attachments/files/18638925/Appendix.1.Full.lists.of.model.response.and.predictors.pdf)


[^4]: Some metrics included both current year and prior year data. Those marked with <sub>[0]</sub> refers to latest/current year data; <sub>[1]</sub> refers to 1 year prior; <sub>[2]</sub> refers to 2 years prior

### Correlation analysis [^5]
![Correlation Matrix](https://github.com/user-attachments/assets/3ad22cc2-6d73-43c7-846f-ab4acf033167)

>**(1)	ROE vs ESG Scores**: ROE % is weakly/moderately positive correlate to ESG combined score, and all individual pillar scores.

>**(2)	ESG Scores vs other financial metrics**:
>
>a.	ESG Combined vs individual pillars: All pillars demonstrate semi-strong to strong positive relations. Governance pillar is relatively weak compared to that of Environmental and Social
>
>b.	For each pillar score, prior year and year before last year have decreasing strength of correlation
>
>c.	Country inflation is moderately negative related to Environmental Score    
>
>d.	Current ration of company is (surprisingly) negatively related to ESG score combined and individually for all pillars scores.   
>
>e.	Net sales are moderately positive correlated to ESG individual pillars score.

### Association analysis
**Chi-square test** was applied to test if some industries are associated with higher or lower ESG score than others. All companies were binned into A, B, C and D grades [^6] according to their ESG combined score. The result p-value is close to zero to reject null hypothesis, i.e. there is **significant association between GICS industries and ESG score**.

[^5]: Strength of correlation is measured with magnitude of correlation coefficient r, which can be described as very strong (|r| >0.8), strong (0.6<|r|<0.8), moderate (0.4<|r|<0.6), weak (0.2<|r|<0.4), and very weak (0<|r|<0.2). 
[^6]: The grading bins in this data analysis: ESG score greater than or equal to 75 as A; Between 50 and 75 as B; Between 25 and 50 as C; Below 25 as D



## Findings
The dissertation employs both parametric (Linear Regression) and non-parametric (Random Forest, Support Vector Regression, Artificial Neural Networks) methods. The models are evaluated based on performance metrics such as adjusted R-squared, mean squared error, and root mean square error. Model findings are summarized below:

![Model Performance Comparison](https://github.com/user-attachments/assets/69671873-fed0-4c09-ac9d-9d070bb7773f)

### Company ESG performance matters in financial outcomes
(1)	Model Selection and performance metric: The analysis reveals that the Random Forest model outperforms other methods, achieving an adjusted R-square of 0.4950. This represents **the best model explained 49.5% of variability of the ROE** with given ESG score and other input characteristics. Linear regression comes 2nd best with adjusted R-square 0.3227.

(2)	Correlation matrix: **ESG scores changes in the same direction with ROE**, but the relationship is not strong. The **weak positive relationship** was supported with correlation coefficient 0.1075 – 0.2008 between ROE, ESG combined scores and individual pillar scores (Correlation coefficient 0 means no linear relation, where one means perfect strong linear relation)

(3)	Feature significance: The features in the best model were measured with their importance with feature importance score ranged from 0 to 1. Closer to 1 represent the feature has greater influence on the model predictions. Profit margin % is the most important feature with score of 0.2814. While ESG combined score measured at 0.0210 compared to same of company net sales 0.0248. **ESG score has comparable significance with annual net sales in ROE prediction**.

(4)	Feature selection: ANOVA F-statistics is 96.26 suggested the **linear model is statistically significant**. With feature selection on all 39 features, **ESG combined score and individual pillar score remained significant predictors** in the final model within selected 23 features.

### ESG impact varied among industries: 
(1)	Feature selection in regression: The dataset contains companies from 74 industries while 17 industries exhibit linear ESG-CFP relationships, with encoded industry features remained in final regression model. **Some industries came with more sophisticated non-linear ESG-CFP relationships** which remained unresolved.

(2)	Feature coefficient: The standardized coefficient of industry exhibited a wide range from -5.0885 to 1.6952. Biotechnology companies had coefficient -5.0885 and capital market companies had 1.6952. If one biotech and one capital market company with same ESG score, the **ROE is more sensitive to company with industry coefficient with higher magnitude**.





## Limitations and Future Implications
**ESG as higher priority for companies**: ROE exhibited moderated positive correlation with ESG scores across the companies. Higher performance in ESG initiatives is associated with higher financial performances.

**Investor shall put focus on company ESG effort**: Company ESG performance shall not be ignored when making investment decisions. ROE is more sensitive in some industries to ESG performance.

**Demand for more comprehensive ESG data**: Future research should aim to expand the dataset to include Asian and African countries to attain a holistic view in the ESG-CFP relationship. Companies’ dataset in the model training was biased to US and European companies due to availability of data in Refinitiv database, which accounted for 79.7% (2,728 out of 3,422) of total observations.  

**Industry specific factors to reveal complex ESG-CFP relationship**: The dissertation serves as a foundational study for understanding the nuanced dynamics between ESG performance and financial outcomes. ESG-CFP relationship was well quantified in 17 industries with the extent ROE would change in respect of ESG scores. Remaining 57 industries were not able to quantify, which call for further industry specific modelling efforts in future.




