## #MakeoverMonday Community Project Activity

[Link to dashboard](https://public.tableau.com/views/WDI_16516753035860/Dashboard?:language=it-IT&publish=yes&:display_count=n&:origin=viz_share_link)

Preview (but check the dashboard to experiment the interaction):

![preview](https://user-images.githubusercontent.com/64132836/166930258-81a48adf-7d2c-44d0-98b2-500bd3ee7b08.jpg)

[Dataset: 2019 Week 10 - World Development Indicators: Health & Equality](https://data.world/makeovermonday/2019w10)

**What works well on the original, and what doesn't?** 
The original Viz is reduced to Africa and it just shows 1 indicator in the tooltip text which shows on moving the mouse over a country.

**How can you make it better?**
1. Introduced the worldmap
2. Chose an indicator of inequality (Gini index)
3. Added Actions to trigger the relevant information about the chosen country:
  a. Table with measures
  b. On selecting a measure, chart with the trend over the period 1960 - 2020

**Challenges/improvements**
I had to workout the original dataset because of the many null values. I did that with Python and Pandas and I save the result to Excel. I narrowed down to the attributes which are more rich of data and I searched for specific attributes of inequality.

I still have limited understanding of the parameters which I can use to control the visualizations. The timeline came up to be an indicator instead of a measured which limited the options to use a line chart.

Thank you!

