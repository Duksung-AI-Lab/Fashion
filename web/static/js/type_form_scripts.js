/*!
* Start Bootstrap - Shop Item v5.0.5 (https://startbootstrap.com/template/shop-item)
* Copyright 2013-2022 Start Bootstrap
* Licensed under MIT (https://github.com/StartBootstrap/startbootstrap-shop-item/blob/master/LICENSE)
*/
// This file is intentionally blank
// Use this file to add JavaScript to your project

function getShirts() {
    // const collar = event.target.value;
    // console.log(collar);





    var collar = document.querySelector('input[name="collars"]:checked').value;
    var pocket = document.querySelector('input[name="pockets"]:checked').value;
    var pattern = document.querySelector('input[name="pattern"]:checked').value;
    var main_color = document.querySelector('input[name="shirts_color"]:checked').value;

    loading("#referimg_shirts");

    $.ajax({
        url: 'test',
        type: 'GET',
        data: {
            'clothes': 'Shirts',
            'collar': collar,
            'pocket': pocket,
            'pattern': pattern,
            'main_color': main_color
        },
        datatype: 'json',
        success: function (data){
            // console.log(data['timestamp'])
            // var element = document.getElementById('result');
            // result.innerText = data['timestamp'];  //test
            if (data['timestamp']) {
                $("#referimg_shirts").attr("src", '/static/images/gan_output/gan_img_outputs_' + data['timestamp'] + '.png');
            }
            closeLoading();
        }
    })




}

function getTshirts() {
    // const collar = event.target.value;
    // console.log(collar);

    var printing = document.querySelector('input[name="printing"]:checked').value;
    var neckline = document.querySelector('input[name="neckline"]:checked').value;
    var sleeve = document.querySelector('input[name="sleeve"]:checked').value;
    var main_color = document.querySelector('input[name="tshirts_color"]:checked').value;

    loading("#referimg_tshirts");


    $.ajax({
        url: 'test',
        type: 'GET',
        data: {
            'clothes': 'Tshirts',
            'printing': printing,
            'neckline': neckline,
            'sleeve': sleeve,
            'main_color': main_color
        },
        datatype: 'json',
        success: function (data){

            // console.log(data['timestamp'])
            // var element = document.getElementById('result');
            // result.innerText = data['timestamp'];  //test
            if (data['timestamp']) {
                $("#referimg_tshirts").attr("src", '/static/images/gan_output/gan_img_outputs_' + data['timestamp'] + '.png');
            }
            closeLoading();
        }
    })


}

function getJeans() {
    // const collar = event.target.value;
    // console.log(collar);

    var fit = document.querySelector('input[name="fit"]:checked').value;
    var washing = document.querySelector('input[name="washing"]:checked').value;
    var damage = document.querySelector('input[name="damage"]:checked').value;
    var color = document.querySelector('input[name="color"]:checked').value;

    loading("#referimg_jeans");


    $.ajax({
        url: 'test',
        type: 'GET',
        data: {
            'clothes': 'Jeans',
            'fit': fit,
            'washing': washing,
            'damage': damage,
            'color': color
        },
        datatype: 'json',
        success: function (data){

            // console.log(data['timestamp'])
            // var element = document.getElementById('result');
            // result.innerText = data['timestamp'];  //test
            if (data['timestamp']) {
                $("#referimg_jeans").attr("src", '/static/images/gan_output/gan_img_outputs_' + data['timestamp'] + '.png');
            }
            closeLoading();
        }
    })


}

function loading(img) {
    // var maskHeight = $(document).height();
    // var maskWidth  = window.document.body.clientWidth;

    var maskHeight = $(img).css("height");
    var maskWidth  = $(img).css("width");
    // var maskx = $("#referimg_shirts").position().left;
    // var masky = $("#referimg_shirts").position().top;
    var off = $(img).offset();

    // var img = document.getElementById("loadingImg");
    // img.style.display = 'block';

    var mask       = "<div id='mask' style='position:absolute; z-index:9000; background-color:#000000; display:none;'></div>";
    // var loadingImg ='';
    // var loadingImg = document.getElementById("loadingImg");


    // loadingImg +=" <div id='loadingImg'>";
    // loadingImg +=" <img src='${pageContext.request.contextPath}/loading.gif' style='position:absolute; z-index:9500; text-align:center; display:block; margin-top:300px; margin-left:750px;'/>";


    // loadingImg +=" <img src=\"{% static \'images/loading.gif\' %}\"  style='position:absolute; z-index:9500; text-align:center; display:block; margin-top:300px; margin-left:750px;'/>";
    // loadingImg += "</div>";

    // console.log(loadingImg);

    $('body')
        .append(mask)

    $('#mask').css({
            'width' : maskWidth,
            'height': maskHeight,
            'left' : off.left,
            'top' : off.top,
            'opacity' :'0.3'
    });

    $('#mask').show();

    // $('.loadingImg').append(loadingImg);
    // $('#loadingImg').show();

    $("#loadingImg").css('display', 'block');


}

function closeLoading() {
    $('#mask, #loadingImg').hide();
    $('#mask').remove();
    // $('#mask, #loadingImg').remove();
    // var img = document.getElementById("loadingImg");
    // img.style.display = 'none';
}