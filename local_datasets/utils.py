def get_create_table(schema_path, db_id):
    try:
        with open(schema_path) as fp:
            flg = False
            #num_column = 0
            create_tables = []
            for line in fp.readlines():
                if "CREATE" in line:
                    flg = True
                    #num_column = line.count('(') - line.count(')')
                    create_tables.append(line)
                elif flg:
                    #num_column = line.count('(') - line.count(')')
                    create_tables[-1] = create_tables[-1] + line
                    if ");" in line: # and num_column == 1:
                        flg = False
                elif flg:
                    create_tables[-1] = create_tables[-1] + line

            return "".join(create_tables)
    except FileNotFoundError:
        print(f"Create table file for {db_id} not found. Use 'custom' schema serialization")
        return ""