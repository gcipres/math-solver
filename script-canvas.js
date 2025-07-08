const clearBtn = document.getElementById('clear');
const checkBtn = document.getElementById('check');

const carryCanvas = document.getElementById('carry-canvas');
const tensCanvas = document.getElementById('tens-canvas');
const unitsCanvas = document.getElementById('units-canvas');

let carry = 0;
let tens = 0;
let units = 0;

let model;
let operation = { operand1: 0, operand2: 0, operator: '+' };
let correctCount = 0;
let totalCount = 0;

// Inicializar canvases como zonas de dibujo
function setupCanvas(canvas) {
  const ctx = canvas.getContext('2d');
  canvas.width = canvas.width; // Clear
  ctx.fillStyle = 'white';
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  let isDrawing = false;
  let hasDrawn = false;

  const getPos = (e) => {
    const rect = canvas.getBoundingClientRect();
    const clientX = e.touches ? e.touches[0].clientX : e.clientX;
    const clientY = e.touches ? e.touches[0].clientY : e.clientY;

    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;

    return {
        x: (clientX - rect.left) * scaleX,
        y: (clientY - rect.top) * scaleY
    };
  };

  const start = (e) => {
    isDrawing = true;

    // Limpiar canvas antes de empezar a dibujar
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    const pos = getPos(e);
    ctx.beginPath();
    ctx.moveTo(pos.x, pos.y);
  };

  const draw = (e) => {
    if (!isDrawing) return;
    hasDrawn = true;

    const pos = getPos(e);
    ctx.lineWidth = 6;
    ctx.lineCap = 'round';
    ctx.strokeStyle = '#000';
    ctx.lineTo(pos.x, pos.y);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(pos.x, pos.y);
  };

  const stop = async(e) => {
    isDrawing = false;
    ctx.beginPath();

    if (!hasDrawn) return;
    hasDrawn = false;

    // Ejecutar predicción al soltar
    if (!model) return;
    const tensor = preprocessCanvas(canvas);
    const feeds = { input: tensor };
    try {
      const output = await model.run(feeds);
      const predictions = output.output.data;
      const predictedNumber = predictions.indexOf(Math.max(...predictions));
      if (e.target.id === 'units-canvas') units = predictedNumber
      if (e.target.id === 'tens-canvas') tens = predictedNumber
      if (e.target.id === 'carry-canvas') carry = predictedNumber

      // Mostrar número sobre el canvas
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.fillStyle = 'white';
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      ctx.fillStyle = '#000';
      ctx.font = 'bold 36px Arial';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(predictedNumber, canvas.width / 2, canvas.height / 2);
    } catch (error) {
      console.error('Error predicting:', error);
    }
  };

  canvas.addEventListener('mousedown', start);
  canvas.addEventListener('mousemove', draw);
  canvas.addEventListener('mouseup', stop);
  canvas.addEventListener('mouseout', stop);
  canvas.addEventListener('touchstart', start);
  canvas.addEventListener('touchmove', draw);
  canvas.addEventListener('touchend', stop);
  canvas.addEventListener('touchcancel', stop);
}

setupCanvas(carryCanvas);
setupCanvas(tensCanvas);
setupCanvas(unitsCanvas);

// Limpiar campos
clearBtn.addEventListener('click', () => {
  [carryCanvas, tensCanvas, unitsCanvas].forEach(c => {
    const ctx = c.getContext('2d');
    ctx.clearRect(0, 0, c.width, c.height);
    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, c.width, c.height);
    c.style.border = '1px solid #ccc';
  });
});

// Generar operación aleatoria
function generateOperation() {
  generateOperationColor();
  carry = 0;
  tens = 0;
  units = 0;

  [carryCanvas, tensCanvas, unitsCanvas].forEach(c => {
    const ctx = c.getContext('2d');
    ctx.clearRect(0, 0, c.width, c.height);
    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, c.width, c.height);
    c.style.border = '1px solid #ccc';
  });

  operation.operand1 = Math.floor(Math.random() * 90 + 10);
  operation.operand2 = Math.floor(Math.random() * 90 + 10);
  operation.operator = Math.random() > 0.5 ? '+' : '-';

  if (operation.operator === '+' && (operation.operand1 + operation.operand2 > 99)) {
    return generateOperation();
  }

  if (operation.operator === '-' && (operation.operand1 < operation.operand2)) {
    return generateOperation();
  }

  updateDisplay();
}

function generateOperationColor() {
  const colors = ["limegreen", "orange", "dodgerblue", "#ba67e4", "#f50e0a", "#fddc8e", "hotpink", "cyan", "magenta", "saddlebrown", "black"];
  const color = colors[Math.floor(Math.random() * colors.length)];

  document.getElementById('operand1').style.color = color;
  document.getElementById('operand2').style.color = color;
  document.getElementById('operator').style.color = color;
  document.getElementById('divider').style.color = color;
  document.getElementById('operation-display').style.borderColor = color;
}

function updateDisplay() {
  document.getElementById('operand1').textContent = operation.operand1;
  document.getElementById('operator').textContent = operation.operator;
  document.getElementById('operand2').textContent = operation.operand2;
}

// Preprocesamiento del canvas a tensor ONNX
function preprocessCanvas(canvas) {
  const ctxTmp = canvas.getContext('2d');
  const width = canvas.width;
  const height = canvas.height;
  const imgData = ctxTmp.getImageData(0, 0, width, height);
  const data = imgData.data;

  let minX = width, minY = height, maxX = 0, maxY = 0;
  let found = false;
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const i = (y * width + x) * 4;
      const r = data[i];
      const a = data[i + 3];
      if (a > 50 && r < 128) {
        found = true;
        if (x < minX) minX = x;
        if (x > maxX) maxX = x;
        if (y < minY) minY = y;
        if (y > maxY) maxY = y;
      }
    }
  }

  if (!found) return new ort.Tensor('float32', new Float32Array(1 * 1 * 28 * 28), [1, 1, 28, 28]);

  const boxW = maxX - minX + 1;
  const boxH = maxY - minY + 1;

  const tmp = document.createElement('canvas');
  tmp.width = 20;
  tmp.height = 20;
  const tmpCtx = tmp.getContext('2d');
  tmpCtx.drawImage(canvas, minX, minY, boxW, boxH, 0, 0, 20, 20);

  const final = document.createElement('canvas');
  final.width = 28;
  final.height = 28;
  const finalCtx = final.getContext('2d');
  finalCtx.fillStyle = 'white';
  finalCtx.fillRect(0, 0, 28, 28);
  finalCtx.drawImage(tmp, 4, 4);

  const img = finalCtx.getImageData(0, 0, 28, 28).data;
  const input = new Float32Array(1 * 1 * 28 * 28);
  for (let i = 0; i < 28 * 28; i++) {
    const r = img[i * 4];
    input[i] = (255 - r) / 255.0;
  }

  return new ort.Tensor('float32', input, [1, 1, 28, 28]);
}

// Validación usando modelo ONNX
async function validateAnswer() {
//   const getPrediction = async (canvas) => {
//     const tensor = preprocessCanvas(canvas);
//     const feeds = { input: tensor };
//     const output = await model.run(feeds);
//     const pred = output.output.data;
//     return pred.indexOf(Math.max(...pred));
//   };

//   const carry = await getPrediction(carryCanvas);
//   const tens = await getPrediction(tensCanvas);
//   const units = await getPrediction(unitsCanvas);

  console.log('carry: ', carry)
  console.log('tens: ', tens)
  console.log('units: ', units)

  // Calcula respuestas correctas
  let correctUnits, correctCarry, correctTens;

  if (operation.operator === '+') {
    correctUnits = (operation.operand1 % 10 + operation.operand2 % 10) % 10;
    correctCarry = Math.floor((operation.operand1 % 10 + operation.operand2 % 10) / 10);
    correctTens = Math.floor(operation.operand1 / 10) + Math.floor(operation.operand2 / 10) + correctCarry;
  } else {
    let units1 = operation.operand1 % 10;
    let units2 = operation.operand2 % 10;
    let tens1 = Math.floor(operation.operand1 / 10);
    let tens2 = Math.floor(operation.operand2 / 10);
    if (units1 < units2) {
      correctUnits = (units1 + 10) - units2;
      correctTens = (tens1 - 1) - tens2;
    } else {
      correctUnits = units1 - units2;
      correctTens = tens1 - tens2;
    }
    correctCarry = units1 < units2 ? 1 : 0;
  }

  // Visual feedback
  carryCanvas.style.border = carry === correctCarry ? '2px solid slategray' : '2px solid red';
  tensCanvas.style.border = tens === correctTens ? '2px solid slategray' : '2px solid red';
  unitsCanvas.style.border = units === correctUnits ? '2px solid slategray' : '2px solid red';

  if (carry === correctCarry && tens === correctTens && units === correctUnits) {
    correctCount++;
    totalCount++;
    document.getElementById('correctAnswers').textContent = correctCount;
    generateOperation();
  }
}

// Cargar modelo ONNX
async function loadModel() {
  model = await ort.InferenceSession.create('digit_classifier.onnx');
  console.log("Modelo cargado.");
}

// Eventos
checkBtn.addEventListener('click', validateAnswer);

// Inicialización
loadModel().then(generateOperation);