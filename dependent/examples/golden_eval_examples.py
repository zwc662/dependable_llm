examples = [
"""
given a database schema( CREATE TABLE IF NOT EXISTS "department" (
"Department_ID" int,
"Name" text,
"Creation" text,
"Ranking" int,
"Budget_in_Billions" real,
"Num_Employees" real,
PRIMARY KEY ("Department_ID")
);
CREATE TABLE IF NOT EXISTS "head" (
"head_ID" int,
"name" text,
"born_state" text,
"age" real,
PRIMARY KEY ("head_ID")
);
CREATE TABLE IF NOT EXISTS "management" (
"department_ID" int,
"head_ID" int,
"temporary_acting" text,
PRIMARY KEY ("Department_ID","head_ID"),
FOREIGN KEY ("Department_ID") REFERENCES `department`("Department_ID"),
FOREIGN KEY ("head_ID") REFERENCES `head`("head_ID")
);
) | wirte a sql script to answer(List the name, born state and age of the heads of departments ordered by age.) | query:
""",
"""
given a database schema( CREATE TABLE IF NOT EXISTS "department" (
"Department_ID" int,
"Name" text,
"Creation" text,
"Ranking" int,
"Budget_in_Billions" real,
"Num_Employees" real,
PRIMARY KEY ("Department_ID")
);
CREATE TABLE IF NOT EXISTS "head" (
"head_ID" int,
"name" text,
"born_state" text,
"age" real,
PRIMARY KEY ("head_ID")
);
CREATE TABLE IF NOT EXISTS "management" (
"department_ID" int,
"head_ID" int,
"temporary_acting" text,
PRIMARY KEY ("Department_ID","head_ID"),
FOREIGN KEY ("Department_ID") REFERENCES `department`("Department_ID"),
FOREIGN KEY ("head_ID") REFERENCES `head`("head_ID")
);
) | wirte a sql script to answer(List the creation year, name and budget of each department.) | query:
""",
"""
given a database schema( CREATE TABLE IF NOT EXISTS "department" (
"Department_ID" int,
"Name" text,
"Creation" text,
"Ranking" int,
"Budget_in_Billions" real,
"Num_Employees" real,
PRIMARY KEY ("Department_ID")
);
CREATE TABLE IF NOT EXISTS "head" (
"head_ID" int,
"name" text,
"born_state" text,
"age" real,
PRIMARY KEY ("head_ID")
);
CREATE TABLE IF NOT EXISTS "management" (
"department_ID" int,
"head_ID" int,
"temporary_acting" text,
PRIMARY KEY ("Department_ID","head_ID"),
FOREIGN KEY ("Department_ID") REFERENCES `department`("Department_ID"),
FOREIGN KEY ("head_ID") REFERENCES `head`("head_ID")
...
);
) | wirte a sql script to answer(What is the average number of employees of the departments whose rank is between 10 and 15?) | query:
""",
"""
Given the schema: CREATE TABLE "entrepreneur" ("Entrepreneur_ID" int, "People_ID" int, "Company" text, "Money_Requested" real,
            "Investor" text, PRIMARY KEY ("Entrepreneur_ID"), FOREIGN KEY ("People_ID") REFERENCES "people"("People_ID")
            ); CREATE TABLE "people" ("People_ID" int, "Name" text,
            "Height" real, "Weight" real, "Date_of_Birth" text, PRIMARY KEY ("People_ID")). Write a sql query to answer the question: "How many entrepreneurs are there?
"""
]