// ======================================
// MODEL PERFORMANCE RADAR CHART
// ======================================

const ctx1 = document.getElementById('performanceChart');

new Chart(ctx1, {
    type: 'radar',

    data: {
        labels: [
            'Accuracy',
            'Precision',
            'Recall',
            'F1-Score',
            'ROC-AUC'
        ],

        datasets: [{
            label: 'AI Performance',

            data: [92, 90, 88, 91, 95],

            backgroundColor: 'rgba(0,255,255,0.2)',
            borderColor: '#00ffff',
            borderWidth: 2,
            pointBackgroundColor: '#00ffff',
            pointBorderColor: '#ffffff',
            pointRadius: 4
        }]
    },

    options: {
        responsive: true,

        plugins: {
            legend: {
                labels: {
                    color: 'white',
                    font: {
                        size: 14
                    }
                }
            }
        },

        scales: {
            r: {
                angleLines: {
                    color: 'rgba(255,255,255,0.2)'
                },

                grid: {
                    color: 'rgba(255,255,255,0.1)'
                },

                pointLabels: {
                    color: 'white',
                    font: {
                        size: 13
                    }
                },

                ticks: {
                    color: 'white',
                    backdropColor: 'transparent'
                },

                suggestedMin: 50,
                suggestedMax: 100
            }
        }
    }
});


// ======================================
// PREDICTION CONFIDENCE LINE CHART
// ======================================

const ctx2 = document.getElementById('confidenceChart');

new Chart(ctx2, {

    type: 'line',

    data: {

        labels: [
            'Jan',
            'Feb',
            'Mar',
            'Apr',
            'May',
            'Jun'
        ],

        datasets: [{
            label: 'Prediction Confidence',

            data: [70, 75, 81, 86, 89, 93],

            borderColor: '#00ff99',
            backgroundColor: 'rgba(0,255,153,0.2)',

            tension: 0.4,
            fill: true,
            borderWidth: 3,
            pointRadius: 5,
            pointBackgroundColor: '#00ff99'
        }]
    },

    options: {

        responsive: true,

        plugins: {
            legend: {
                labels: {
                    color: 'white',
                    font: {
                        size: 14
                    }
                }
            }
        },

        scales: {

            x: {
                ticks: {
                    color: 'white'
                },

                grid: {
                    color: 'rgba(255,255,255,0.1)'
                }
            },

            y: {

                ticks: {
                    color: 'white'
                },

                grid: {
                    color: 'rgba(255,255,255,0.1)'
                },

                beginAtZero: true,
                max: 100
            }
        }
    }
});


// ======================================
// RISK DISTRIBUTION DOUGHNUT CHART
// ======================================

const ctx3 = document.getElementById('riskChart');

new Chart(ctx3, {

    type: 'doughnut',

    data: {

        labels: [
            'High Risk',
            'Low Risk'
        ],

        datasets: [{
            data: [35, 65],

            backgroundColor: [
                '#ff4d6d',
                '#00ff99'
            ],

            borderWidth: 0
        }]
    },

    options: {

        responsive: true,

        plugins: {
            legend: {
                labels: {
                    color: 'white',
                    font: {
                        size: 14
                    }
                }
            }
        }
    }
});


// ======================================
// BMI DISTRIBUTION BAR CHART
// ======================================

const ctx4 = document.getElementById('bmiChart');

new Chart(ctx4, {

    type: 'bar',

    data: {

        labels: [
            'Underweight',
            'Normal',
            'Overweight',
            'Obese'
        ],

        datasets: [{
            label: 'Patients',

            data: [12, 45, 28, 15],

            backgroundColor: [
                '#00ffff',
                '#00ff99',
                '#ffaa00',
                '#ff4d6d'
            ],

            borderRadius: 10
        }]
    },

    options: {

        responsive: true,

        plugins: {
            legend: {
                labels: {
                    color: 'white'
                }
            }
        },

        scales: {

            x: {
                ticks: {
                    color: 'white'
                },

                grid: {
                    color: 'rgba(255,255,255,0.05)'
                }
            },

            y: {
                ticks: {
                    color: 'white'
                },

                grid: {
                    color: 'rgba(255,255,255,0.05)'
                },

                beginAtZero: true
            }
        }
    }
});