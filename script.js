const canvas = document.getElementById('draw-area');
const clearBtn = document.getElementById('clear');
const checkBtn = document.getElementById('check');
const message = document.getElementById('message');
const ctx = canvas.getContext('2d');

let model;
let isDrawing = false;
let operation = { operand1: 0, operand2: 0, operator: '+' };
let answer = 0;
let correctCount = 0;
let totalCount = 0;

// Configuración del canvas
function setupCanvas() {
    canvas.width = 350;
    canvas.height = 200;
    setupEventListeners();
}

// Eventos del canvas
function setupEventListeners() {
    canvas.addEventListener('mousedown', startDrawing);
    canvas.addEventListener('mousemove', draw);
    canvas.addEventListener('mouseup', stopDrawing);
    canvas.addEventListener('mouseout', stopDrawing);

    canvas.addEventListener('touchstart', (event) => {
        event.preventDefault();
        startDrawing(event);
    });
    canvas.addEventListener('touchmove', (event) => {
        event.preventDefault();
        draw(event);
    });
    canvas.addEventListener('touchend', stopDrawing);
    canvas.addEventListener('touchcancel', stopDrawing);

    clearBtn.addEventListener('click', clearCanvas);
    checkBtn.addEventListener('click', checkAnswer);
}

// Funciones de dibujo
function getPointerPosition(event) {
    if (event.touches) {
        const touch = event.touches[0];
        const rect = canvas.getBoundingClientRect();
        return {
            x: touch.clientX - rect.left,
            y: touch.clientY - rect.top
        };
    } else {
        return { x: event.offsetX, y: event.offsetY };
    }
}

function draw(event) {
    if (!isDrawing) return;

    const pos = getPointerPosition(event);
    ctx.lineWidth = 10;
    ctx.lineCap = 'round';
    ctx.strokeStyle = '#000';

    ctx.lineTo(pos.x, pos.y);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(pos.x, pos.y);
}

function startDrawing(event) {
    isDrawing = true;
    const pos = getPointerPosition(event);
    ctx.beginPath();
    ctx.moveTo(pos.x, pos.y);
}

function stopDrawing() {
    isDrawing = false;
    ctx.beginPath();
}

function clearCanvas() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
}

// Lógica del juego
function generateOperation() {
    operation.operand1 = Math.floor(Math.random() * 90 + 10);
    operation.operand2 = Math.floor(Math.random() * 90 + 10);
    operation.operator = Math.random() > 0.5 ? '+' : '-';

    if (operation.operator === '+' && (operation.operand1 + operation.operand2 > 99)) {
        generateOperation();
    }

    if (operation.operator === '-' && (operation.operand1 < operation.operand2)) {
        generateOperation();
    }

    answer = operation.operator === '+' ? operation.operand1 + operation.operand2 : operation.operand1 - operation.operand2;
    updateDisplay();
}

function updateDisplay() {
    document.getElementById('operand1').textContent = operation.operand1;
    document.getElementById('operator').textContent = operation.operator;
    document.getElementById('operand2').textContent = operation.operand2;
}

// Modelo de TensorFlow.js
async function loadModel() {
    try {
        model = await tf.loadLayersModel('./tmp/tfjs_model/model.json');
        console.log('Model loaded');
    } catch (error) {
        console.error('Error loading model:', error);
        message.textContent = 'Error loading model. Please try again.';
        message.style.color = 'red';
    }
}

// Validación de la respuesta
async function checkAnswer() {
    if (!model) {
        message.textContent = 'Model not loaded yet.';
        message.style.color = 'red';
        return;
    }

    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const tensor = tf.browser.fromPixels(imageData, 1)
    .resizeNearestNeighbor([28, 28])
    .toFloat()
    .div(255.0)
    .mul(-1)
    .add(1)
    .squeeze([-1])   // Quita la dimensión extra
    .expandDims(0);  // ✅ Resultado: (1, 28, 28)

    console.log("Tensor shape:", tensor.shape); // Debe ser [1, 28, 28]

    try {
        const prediction = model.predict(tensor);
        const predictedNumber = prediction.argMax(1).dataSync()[0];

        console.log(predictedNumber, answer);

        if (predictedNumber === answer) {
            clearCanvas();
            canvas.style.border = '2px solid black';
            correctCount++;
            message.textContent = 'Correct!';
            message.style.color = 'green';
            generateOperation();
        } else {
            clearCanvas();
            canvas.style.border = '2px solid red';
            message.textContent = 'Incorrect!';
            message.style.color = 'red';
        }
        totalCount++;
    } catch (error) {
        console.error('Error predicting:', error);
        message.textContent = 'Error predicting. Please try again.';
        message.style.color = 'red';
    }
}

// Inicialización
async function initialize() {
    setupCanvas();
    generateOperation();
    await loadModel();
}

initialize();