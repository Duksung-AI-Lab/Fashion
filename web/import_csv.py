import csv
import os
import django

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
django.setup()


from select_type.models import Shirts, Jeans, Tshirts

# records = Shirts.objects.all()
# records.delete()
#
# with open('static/csv/Shirts_DB_hue.csv', encoding='utf-8') as csv_file_sub_categories:
#     rows = csv.reader(csv_file_sub_categories)
#     next(rows, None)
#
#     for row in rows:
#         Shirts.objects.create(
#             name=row[0],
#             filename=row[1],
#             collar=row[2],
#             pattern=row[3],
#             pocket=row[4],
#             main_color=row[5],
#             sub_color=row[6],
#             price=row[7],
#             link=row[8],
#             main_color_h=row[9],
#             sub_color_h=row[10]
#         )
#         print(row)


records = Tshirts.objects.all()
records.delete()

with open('static/csv/Tshirts_DB_hue.csv', encoding='utf-8') as csv_file_sub_categories:
    rows = csv.reader(csv_file_sub_categories)
    next(rows, None)

    for row in rows:
        Tshirts.objects.create(
            name=row[0],
            filename=row[1],
            price=row[2],
            link=row[3],
            printing=row[4],
            sleeve=row[5],
            neckline=row[6],
            main_color=row[7],
            sub_color=row[8],
            main_color_h=row[9],
            sub_color_h=row[10]
        )
        print(row)


# records = Jeans.objects.all()
# records.delete()
#
# with open('static/csv/Jeans_DB.csv', encoding='utf-8') as csv_file_sub_categories:
#     rows = csv.reader(csv_file_sub_categories)
#     next(rows, None)
#
#     for row in rows:
#         Jeans.objects.create(
#             name=row[0],
#             filename=row[1],
#             fit=row[2],
#             color=row[3],
#             washing=row[4],
#             damage=row[5],
#             price=row[6],
#             link=row[7]
#         )
#         print(row)
