{{ config(materialized='table') }}

select
    education,
    avg(case when income = '>50K' then 1.0 else 0.0 end) as pct_high_earners,
    avg(age) as avg_age,
    avg(hours_per_week) as avg_hours,
    count(*) as n_people
from {{ ref('stg_adult') }}
group by education
order by pct_high_earners desc