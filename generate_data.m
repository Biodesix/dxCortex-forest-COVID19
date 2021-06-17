cont_ftrs = {
    'ed_temperature_c',
    'ed_heart_rate',
    'ed_systolic_bp',
    'ed_diastolic_bp',
    'ed_respiratory_rate',
    'ed_oxygen_saturation',
    'initial_qtc',
    'sodium',
    'potassium',
    'carbon_dioxide_bicarb',
    'bun',
    'creatinine',
    'anion_gap',
    'wbc_screen',
    'hemoglobin',
    'hematocrit',
    'platelet_count',
    'initial_ldh',
    'initial_d_dimer',
    'initial_c_reactive_protein',
    'ferritin',
    'age',
    'weight_kg'
};

cat_ftrs = {
    'Race',
    'sex_or_gender',
    'egfr'
};


dg = data_generator('./data/FT.csv', cont_ftrs, cat_ftrs, 'Definition');
dg.generate(200, './data/sythetic_training.csv');
dg.generate(100, './data/sythetic_validation.csv');
