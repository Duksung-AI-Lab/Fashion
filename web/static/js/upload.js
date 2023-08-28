var otherTab = document.getElementById('select-navi')
otherTab.ontouchend = TouchTab

function showImage(input) {
    var file = input.files[0];

    var prevImage = document.getElementById('image-show').lastElementChild;
        if(prevImage) {prevImage.remove();}

    var name = document.getElementById('fileName');
    name.textContent = file.name;

    var newImage = document.createElement("img");
    newImage.setAttribute("class", 'img');

    newImage.src = URL.createObjectURL(file);

    newImage.style.width = "200px";
    newImage.style.height = "auto";
    newImage.style.objectFit = "auto";

    var container = document.getElementById('image-show');
    container.appendChild(newImage);


    //document.getElementById('fileName').textContent = null;     //기존 파일 이름 지우기
}

function TouchTab(){

    var prevImage = document.getElementById('image-show').lastElementChild;
        if(prevImage) {prevImage.remove();}

    document.getElementById('fileName').textContent = null;     //기존 파일 이름 지우기
}
