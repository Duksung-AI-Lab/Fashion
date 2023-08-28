from django.db import models


class Shirts(models.Model):
    name = models.CharField(max_length=100)
    filename = models.CharField(max_length=50)
    collar = models.CharField(max_length=10)
    pattern = models.CharField(max_length=10)
    pocket = models.CharField(max_length=5)
    main_color = models.CharField(max_length=10, default='', null=True)
    sub_color = models.CharField(max_length=10, default='', null=True)
    price = models.CharField(max_length=10)
    link = models.CharField(max_length=100)
    main_color_h = models.CharField(max_length=20, default='', null=True)
    sub_color_h = models.CharField(max_length=20, default='', null=True)

    class Meta:
        db_table = 'shirts'


class Tshirts(models.Model):
    name = models.CharField(max_length=100)
    filename = models.CharField(max_length=50)
    price = models.CharField(max_length=10)
    link = models.CharField(max_length=100)
    printing = models.CharField(max_length=10)
    sleeve = models.CharField(max_length=10)
    neckline = models.CharField(max_length=10)
    main_color = models.CharField(max_length=10, default='', null=True)
    sub_color = models.CharField(max_length=10, default='', null=True)
    main_color_h = models.CharField(max_length=20, default='', null=True)
    sub_color_h = models.CharField(max_length=20, default='', null=True)

    class Meta:
        db_table = 'Tshirts'


class Jeans(models.Model):
    name = models.CharField(max_length=100)
    filename = models.CharField(max_length=50)
    fit = models.CharField(max_length=10)
    color = models.CharField(max_length=10)
    washing = models.CharField(max_length=10)
    damage = models.CharField(max_length=10)
    price = models.CharField(max_length=10)
    link = models.CharField(max_length=100)

    class Meta:
        db_table = 'Jeans'
