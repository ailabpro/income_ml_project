{{ config(materialized='view') }}

select
    age,
    workclass,
    education,
    education_num,
    marital_status,
    occupation,
    relationship,
    race,
    sex,
    capital_gain,
    capital_loss,
    hours_per_week,
    native_country,
    trim(income) as income
from {{ source('raw', 'raw_adult') }}
where workclass is not null
  and occupation is not null