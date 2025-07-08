const container = document.getElementById('container');
const clearBtn = document.getElementById('clear');
const checkBtn = document.getElementById('check');
const carryInput = document.getElementById('carry');
const tensInput = document.getElementById('tens');  
const unitsInput = document.getElementById('units');

let operation = { operand1: 0, operand2: 0, operator: '+' };
let correctCount = 0;
let totalCount = 0;

clearBtn.addEventListener('click', () => {
  carryInput.value = '';
  tensInput.value = '';
  unitsInput.value = '';
});

function generateOperation() {
  generateOperationColor();

  carryInput.value = '';
  tensInput.value = '';
  unitsInput.value = '';

  operation.operand1 = Math.floor(Math.random() * 90 + 10);
  operation.operand2 = Math.floor(Math.random() * 90 + 10);
  operation.operator = Math.random() > 0.5 ? '+' : '-';

  if (operation.operator === '+' && (operation.operand1 + operation.operand2 > 99)) {
    generateOperation();
  } 
  
  if (operation.operator === '-' && (operation.operand1 < operation.operand2)) {
    generateOperation();
  }

  updateDisplay();
}

function generateOperationColor() {
  //green orange blue purple red yellow pink cyan magenta brown black
  const basicColors = ["limegreen", "orange", "dodgerblue", "#ba67e4", "#f50e0a", "#fddc8e", "hotpink", "cyan", "magenta", "saddlebrown", "black"];
  const color = basicColors[Math.floor(Math.random() * basicColors.length)];

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

// Validar la respuesta ingresada por el usuario
function validateAnswer() {
  let carry = parseInt(carryInput.value) || 0;
  let tens = parseInt(tensInput.value) || 0;
  let units = parseInt(unitsInput.value) || 0;

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

  document.getElementById("carry").style.border = carry === correctCarry ? '2px solid slategray' : '2px solid red';
  document.getElementById("tens").style.border = tens === correctTens ? '2px solid slategray' : '2px solid red';
  document.getElementById("units").style.border = units === correctUnits ? '2px solid slategray' : '2px solid red';

  if (carry === correctCarry && tens === correctTens && units === correctUnits) {
    correctCount++;
    totalCount++;
    document.getElementById('correctAnswers').textContent = correctCount;
    generateOperation();
  }
}

// Evento para verificar la respuesta cuando el usuario hace clic en "Check"
checkBtn.addEventListener('click', () => {
  validateAnswer();
});

// Inicialización
generateOperation();
