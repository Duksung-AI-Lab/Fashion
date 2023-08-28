import time
from django.shortcuts import render
from . import clothes_predict
from django.db import connection
from django.core.files.storage import FileSystemStorage  # 파일저장


def index(request):
    return render(request, 'upload_image/upload.html')


def post_clothes(request):
    timestamp = time.strftime('%Y:%m:%d-%H:%M:%S')

    if request.method == 'POST':
        search_item_type = request.POST['search_item_type']

        # Save User Image
        user_img = request.FILES['user_img']
        img_path = 'user_img_' + timestamp + '.jpg'
        fs = FileSystemStorage()
        fs.save(img_path, user_img)
        # user_img.save(img_path)
        print("REFER_IMG:", img_path)

        if search_item_type == 'shirts':

            t = time.time()
            collars, pattern, pockets, main_color, sub_color = clothes_predict.shirts_predict(timestamp)
            print('predict takes ', (time.time() - t), 'secs')
            print(collars, pattern, pockets, main_color, sub_color)

            sql_filter = []

            try:
                cursor = connection.cursor()

                strSql = "SELECT name, filename, price, link FROM shirts"

                if collars is not None:
                    sql_filter.append(f"collar = \"{collars}\"")
                if pockets is not None:
                    sql_filter.append(f"pocket = \"{pockets}\"")
                if pattern is not None:
                    sql_filter.append(f"pattern = \"{pattern}\"")
                if main_color is not None:
                    sql_filter.append(f"main_color_h = \"{main_color}\"")
                if sub_color is not None:
                    sql_filter.append(f"sub_color_h = \"{sub_color}\"")

                if sql_filter:
                    strSql += " WHERE "
                    strSql += ' and '.join(sql_filter)
                else:
                    strSql = "SELECT name, filename, price, link FROM shirts WHERE collar=\"none\""

                print(strSql)

                result = cursor.execute(strSql, )
                datas = cursor.fetchall()

                shirts = []
                for data in datas:
                    row = {'name': data[0],
                           'filename': data[1],
                           'price': data[2],
                           'link': data[3]
                           }

                    shirts.append(row)

                connection.commit()
                connection.close()

                context = {'shirts': shirts}

            except:
                connection.rollback()
                print("Failed selecting in Shirts")

        if search_item_type == 'tshirts':

            t = time.time()
            printing, sleeve, neckline, main_color, sub_color = clothes_predict.tshirts_predict(timestamp)
            print('predict takes ', (time.time() - t), 'secs')
            print(printing, sleeve, neckline, main_color, sub_color)

            sql_filter = []

            try:
                cursor = connection.cursor()

                strSql = "SELECT name, filename, price, link FROM Tshirts"

                if printing is not None:
                    sql_filter.append(f"printing = \"{printing}\"")
                if sleeve is not None:
                    sql_filter.append(f"sleeve = \"{sleeve}\"")
                if neckline is not None:
                    sql_filter.append(f"neckline = \"{neckline}\"")
                if main_color is not None:
                    sql_filter.append(f"main_color_h = \"{main_color}\"")
                if sub_color is not None:
                    sql_filter.append(f"sub_color_h = \"{sub_color}\"")

                if sql_filter:
                    strSql += " WHERE "
                    strSql += ' and '.join(sql_filter)
                else:
                    strSql = "SELECT name, filename, price, link FROM Tshirts WHERE printing=\"none\""

                print(strSql)

                result = cursor.execute(strSql, )
                datas = cursor.fetchall()

                tshirts = []
                for data in datas:
                    row = {'name': data[0],
                           'filename': data[1],
                           'price': data[2],
                           'link': data[3]
                           }

                    tshirts.append(row)

                connection.commit()
                connection.close()

                context = {'tshirts': tshirts}

            except:
                connection.rollback()
                print("Failed selecting in Tshirts")

        if search_item_type == 'jeans':

            t = time.time()
            fit, damage, washing, color = clothes_predict.jeans_predict(timestamp)
            print('predict takes ', (time.time() - t), 'secs')
            print(fit, damage, washing, color)

            sql_filter = []

            try:
                cursor = connection.cursor()

                strSql = "SELECT name, filename, price, link FROM Jeans"

                if fit is not None:
                    sql_filter.append(f"fit = \"{fit}\"")
                if damage is not None:
                    sql_filter.append(f"damage = \"{damage}\"")
                if washing is not None:
                    sql_filter.append(f"washing = \"{washing}\"")
                if color is not None:
                    sql_filter.append(f"color = \"{color}\"")

                if sql_filter:
                    strSql += " WHERE "
                    strSql += ' and '.join(sql_filter)
                else:
                    strSql = "SELECT name, filename, price, link FROM shirts WHERE fit=\"none\""

                print(strSql)

                result = cursor.execute(strSql, )
                datas = cursor.fetchall()

                jeans = []
                for data in datas:
                    row = {'name': data[0],
                           'filename': data[1],
                           'price': data[2],
                           'link': data[3]
                           }

                    jeans.append(row)

                connection.commit()
                connection.close()

                context = {'jeans': jeans}

            except:
                connection.rollback()
                print("Failed selecting in Jeans")

    return render(request, 'result.html', context)



