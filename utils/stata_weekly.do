import delimited INPUT_FILENAME, delimiter(comma) varnames(1)

gen year = real(substr(date, 1, 4))
gen month = real(substr(date, 6, 2))
gen day = real(substr(date, 9, 2))

drop date
gen date = mdy(month, day, year)
format date %tdCCYY.NN.DD
drop year month day

gen week_of = cond(dow(date) == 0, date, date - dow(date))
format week_of %tdCCYY.NN.DD

bysort week_of: egen calc_weekly = mean(raw_weekly)

gen weight = raw_weekly / calc_weekly

gen weighted_weekly = raw_weekly * weight

egen max_weighted_weekly = max(weighted_weekly)

gen harmonized_weekly = (weighted_weekly / max_weighted_weekly) * 100

drop week_of calc_weekly weight weighted_weekly max_weighted_weekly

order date

export delimited OUTPUT_FILENAME
