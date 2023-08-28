import time
import traceback

from django.http import JsonResponse
from django.shortcuts import render
from django.db import connection
from .pix2pix_clothes import exe_pix2pix


def index(request):
    return render(request, 'select_type/type_form.html')


def get_ajax(request):
    clothes = request.GET.get('clothes')
    if clothes == 'Shirts':
        collar = request.GET.get('collar')
        pocket = request.GET.get('pocket')
        pattern = request.GET.get('pattern')
        color = request.GET.get('main_color')

        if collar == 'none':
            collar = 'Regular'
        if pocket == 'none':
            pocket = '0'
        if pattern == 'none':
            pattern = 'solid'
        if color == 'none':
            color = 'White'

        path = f"{clothes}/{collar}_{pattern}_{pocket}/{color}/*"

    if clothes == 'Tshirts':
        printing = request.GET.get('printing')
        neckline = request.GET.get('neckline')
        sleeve = request.GET.get('sleeve')
        color = request.GET.get('main_color')

        if printing == 'none':
            printing = 'plain'
        if neckline == 'none':
            neckline = 'round'
        if sleeve == 'none':
            sleeve = 'half'
        if color == 'none':
            color = 'White'

        path = f"{clothes}/{sleeve}/{neckline}/{printing}/{color}/*"

    if clothes == 'Jeans':
        fit = request.GET.get('fit')
        washing = request.GET.get('washing')
        damage = request.GET.get('damage')
        color = request.GET.get('color')

        if fit == 'none':
            fit = 'regular'
        if washing == 'none':
            washing = 'one'
        if damage == 'none':
            damage = 'nondamage'
        if color == 'none':
            color = 'LightTone'

        path = f"{clothes}/{fit}_{damage}_{washing}_{color}/*"

    # print(path)
    timestamp = time.strftime('%Y:%m:%d-%H:%M:%S')
    istrue = exe_pix2pix(path, timestamp)
    if istrue:
        context = {'timestamp': timestamp}
    else:
        context = {'timestamp': None}
    return JsonResponse(context)



def post_shirts(request):  ## 셔츠 검색

    if request.method == 'POST':
        collars = request.POST['collars']
        pockets = request.POST['pockets']
        pattern = request.POST['pattern']
        main_color = request.POST['shirts_color']
        # sub_color = request.POST['shirts_color_sub']

        sql_filter = []
        item_filter = []

        try:
            cursor = connection.cursor()

            strSql = "SELECT name, filename, price, link FROM shirts"

            if collars != "none":
                sql_filter.append(f"collar = \"{collars}\"")
                item_filter.append(f"collars : {collars}")
            if pockets != "none":
                sql_filter.append(f"pocket = \"{pockets}\"")
                item_filter.append(f"pockets : {pockets}")
            if pattern != "none":
                sql_filter.append(f"pattern = \"{pattern}\"")
                item_filter.append(f"pattern : {pattern}")
            if main_color != "none":
                sql_filter.append(f"main_color_h = \"{main_color}\"")
                item_filter.append(f"main color : {main_color}")
            # if sub_color != "none":
            #     sql_filter.append(f"sub_color_h = \"{sub_color}\"")
            #     item_filter.append(f"sub color : {sub_color}")

            if sql_filter:
                strSql += " WHERE "
                strSql += ' and '.join(sql_filter)

            print(strSql)

            result = cursor.execute(strSql,)
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

        except Exception as e:
            connection.rollback()
            print("Failed selecting in Shirts\n")
            print(traceback.format_exc())

    return render(request, 'result.html', {'shirts': shirts, 'filter': item_filter})


def post_tshirts(request):  ## 티셔츠 검색

    if request.method == 'POST':
        printing = request.POST['printing']
        sleeve = request.POST['sleeve']
        neckline = request.POST['neckline']
        main_color = request.POST['tshirts_color']
        # sub_color = request.POST['tshirts_color_sub']

        sql_filter = []
        item_filter = []

        try:
            cursor = connection.cursor()

            strSql = "SELECT name, filename, price, link FROM Tshirts"

            if printing != "none":
                sql_filter.append(f"printing = \"{printing}\"")
                item_filter.append(f"printing : {printing}")
            if sleeve != "none":
                sql_filter.append(f"sleeve = \"{sleeve}\"")
                item_filter.append(f"sleeve : {sleeve}")
            if neckline != "none":
                sql_filter.append(f"neckline = \"{neckline}\"")
                item_filter.append(f"neck line : {neckline}")
            if main_color != "none":
                sql_filter.append(f"main_color_h = \"{main_color}\"")
                item_filter.append(f"main color : {main_color}")
            # if sub_color != "none":
            #     sql_filter.append(f"sub_color_h = \"{sub_color}\"")
            #     item_filter.append(f"sub color : {sub_color}")

            if sql_filter:
                strSql += " WHERE "
                strSql += ' and '.join(sql_filter)

            print(strSql)

            result = cursor.execute(strSql,)
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

            # output_dir = pix2pix_clothes.exe_pix2pix(collars)
            # output_path = output_dir + "regular_stripe_0_blue-outputs.png"

        except:
            connection.rollback()
            print("Failed selecting in Tshirts")

    return render(request, 'result.html', {'tshirts': tshirts, 'filter': item_filter})


def post_jeans(request):  ## 청바지 검색
    jeans = []
    if request.method == 'POST':
        fit = request.POST['fit']
        washing = request.POST['washing']
        damage = request.POST['damage']
        color = request.POST['color']

        sql_filter = []
        item_filter = []

        try:
            cursor = connection.cursor()

            strSql = "SELECT name, filename, price, link FROM Jeans"

            if fit != "none":
                sql_filter.append(f"fit = \"{fit}\"")
                item_filter.append(f"fit : {fit}")
            if washing != "none":
                sql_filter.append(f"washing = \"{washing}\"")
                item_filter.append(f"washing : {washing}")
            if damage != "none":
                sql_filter.append(f"damage = \"{damage}\"")
                item_filter.append(f"damage : {damage}")
            if color != "none":
                sql_filter.append(f"color = \"{color}\"")
                item_filter.append(f"color : {color}")

            if sql_filter:
                strSql += " WHERE "
                strSql += ' and '.join(sql_filter)

            print(strSql)

            result = cursor.execute(strSql,)
            datas = cursor.fetchall()

            for data in datas:
                row = {'name': data[0],
                       'filename': data[1],
                       'price': data[2],
                       'link': data[3]
                       }

                jeans.append(row)

            connection.commit()
            connection.close()

        except:
            connection.rollback()
            print("Failed selecting in Jeans")

    return render(request, 'result.html', {'jeans': jeans, 'filter': item_filter})
