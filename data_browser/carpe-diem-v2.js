let data;

const skipColumns = {
    umap1: true, umap2: true, icu_rank: true, icu_day: true, discharge_disposition_name: true,
    pt_study_id: true
};

const dataColumns = [
    'SOFA_score', 
    // Intubated
    'Intubation_flag', 
    // Sedation
    'RASS_score',
    'GCS_eye_opening', 'GCS_motor_response', 'GCS_verbal_response',
    // Lung injury severity
    'PaO2FIO2_ratio', 'ABG_PaO2', 
    'FiO2', 'PEEP', 'Oxygen_saturation', 
    'Plateau_Pressure', 'Lung_Compliance', 'ECMO_flag',
    // Hemodynamics/shock
    'Norepinephrine_flag', 
    'Norepinephrine_rate', 'Mean_arterial_pressure',
    'Systolic_blood_pressure',
    'Diastolic_blood_pressure',
    'Lactic_acid', 'Hemoglobin', 
    'ABG_pH', 
    'ABG_PaCO2', 
    'Bicarbonate', 
    // Renal
    'CRRT_flag', 'Hemodialysis_flag', 'Creatinine', 'Urine_output', 
    // Inflammatory biomarkers
    'WBC_count', 'Neutrophils', 'Platelets', 'Procalcitonin', 'CRP',
    'D_dimer', 'LDH', 'Ferritin', 
    'Bilirubin',  'Albumin', 'Lymphocytes',
    // Vitals
    'Temperature', 
    'Heart_rate', 
    'Respiratory_rate',
    // Instability
    'Respiratory_rate_changes', 'PEEP_changes', 'FiO2_changes'
];

const columnNames = {
    "Temperature": "Mean_temperature",
    "Median_temperature": "Med_temperature",
    "Heart_rate": "Mean_heart_rate",
    "Median_heart_rate": "Med_heart_rate",
    "Systolic_blood_pressure": "Mean_SBP",
    "Median_systolic_blood_pressure": "Med_SBP",
    'Max_systolic_blood_pressure': "Max_SBP",
    'Diastolic_blood_pressure': "Mean_DBP", 
    'Median_diastolic_blood_pressure': "Med_DBP",
    'Max_diastolic_blood_pressure': "Max_DBP", 
    'Mean_arterial_pressure': "MAP",
    'Median_respiratory_rate': "Med_respiratory_rate",
    'Oxygen_saturation': "Mean_O2_saturation",
    'Min_oxygen_saturation': "Min_O2_saturation", 
    'Median_oxygen_saturation': "Med_O2_saturation",
    'Max_oxygen_saturation': "Max_O2_saturation",
    'Respiratory_rate_changes': 'Resp_rate_changes'
};

let tab20 = [ // reordered matplotlib tab20
    "#1f77b4",
    "#aec7e8",
    "#ff7f0e",
    "#ffbb78",
    "#2ca02c",
    "#98df8a",
    "#d62728",
    "#ff9896",
    "#9467bd",
    "#c5b0d5",
    "#8c564b",
    "#c49c94",
    "#e377c2",
    "#f7b6d2",
    "#7f7f7f",
    "#c7c7c7",
    "#bcbd22",
    "#dbdb8d",
    "#17becf",
    "#9edae5"
];

const DISCHARGE_COLORS = [
    ["Home", "#08a3e090"],
    ["Rehab", "#085fe090"],
    ["SNF", "#4b3cbc90"],
    ["LTACH", "#7e26b590"],
    ["Hospice", "#aa1b4190"],
    ["Died", "#e00a0690"]
];

let umapMargin = {top: 0, right: 30, bottom: 40, left: 40};
let umapWidth = 500 - umapMargin.left - umapMargin.right;
let umapHeight = 500 - umapMargin.top - umapMargin.bottom;

let margin = {top: 30, right: 30, bottom: 40, left: 40};
let width = 400 - margin.left - margin.right;
let height = 150 - margin.top - margin.bottom;

let histMargin = {top: 15, right: 10, bottom: 18, left: 10};
let histWidth = 150 - histMargin.left - histMargin.right;
let histHeight = 50 - histMargin.top - histMargin.bottom;

let dischargeColor;
let selected = false;
let showingTrends = false;

let umap;
let umapX;
let umapY;

let trajectoriesX;
let trajectoryY;

let currentPoint;

let hists = {};
let distributionX = {};
let distributionRanges = {};

const colorUmap = function(column) {
    let gradient = ["#ffcbad", "#6b2904"];

    d3.select(".sofa-scale").classed("invisible", true);

    let colorScale;
    if (column == "clusters") {
        colorScale = d3.scaleOrdinal()
            .domain(Object.keys(tab20))
            .range(tab20);
    } else {
        d3.select(".sofa-scale")
            .classed("invisible", false)
            .style("width", umapWidth + "px")
            .style("margin-left", margin.left + "px")
            .style("background",
                "linear-gradient(-90deg, " + gradient[1] + ", " + gradient[0] + ")");

        let scale = distributionX[column];
        let range = distributionRanges[column];
        let ticks = scale.ticks(5);
        let tickPositions = [];
        ticks.forEach(function(tick) {
            let tickPos = (tick - range[0]) / (range[1] - range[0]) * umapWidth;
            tickPositions.push(tickPos);
        });
        colorScale = d3.scaleLinear()
            .domain(range)
            .range(gradient);

        d3.selectAll(".sofa-scale div").remove();
        d3.select(".sofa-scale")
            .selectAll("tick")
            .data(d3.zip(ticks, tickPositions))
            .enter()
            .append("div")
                .style("position", "absolute")
                .style("top", "0px")
                .style("left", function(x) { return x[1] - 1 + "px" })
                .style("height", "12px")
                .style("width", "1px")
                .style("background", "#333")
            .append("span")
                .text(function(x) { return x[0] })
                .style("position", "absolute")
                .style("top", "14px")
                .style("left", "0px")
                .style("transform", "translateX(-50%)");
    }

    let dataSorted = data.slice().sort(function(a, b) {
        return Math.abs(a[column]) - Math.abs(b[column]);
    });
    umap.selectAll(".umap-dots").remove();

    let hasNA = false;
    umap.append("g")
        .classed("umap-dots", true)
        .selectAll("dot")
        .data(dataSorted)
        .enter()
        .append("circle")
        .attr("cx", function(d) { return umapX(+d.umap1); } )
        .attr("cy", function(d) { return umapY(+d.umap2); } )
        .attr("r", 2)
        .style("fill", function(d) {
            if (d[column] == "") {
                hasNA = true;
                return "#a9c4d3";
            }
            return colorScale(+d[column])
        } );

    umap.select(".colored-by").remove();
    umap.append("text")
        .classed("colored-by", true)
        .attr("text-anchor", "start")
        .attr("y", umapHeight - 5)
        .attr("x", 5)
        .text("Color: " + column);

    umap.selectAll(".na-color").remove();
    if (hasNA) {
        umap.append("text")
            .classed("na-color", true)
            .attr("text-anchor", "start")
            .attr("x", umapWidth - 30)
            .attr("y", umapHeight + 34)
            .text("NA");
        umap.append("rect")
            .classed("na-color", true)
            .attr("x", umapWidth - 50.5)
            .attr("y", umapHeight + 20.5)
            .attr("width", 15)
            .attr("height", 15)
            .attr("stroke", "#333333")
            .attr("stroke-width", 1)
            .attr("fill", "#a9c4d3");
    }

    d3.select(".color-reset").classed("invisible", column == "clusters");

    umap.selectAll(".bal-overlay").raise();
    umap.selectAll(".patient-overlay").raise();
};

const showBALs = function() {
    let dataBAL = data.slice().filter(function(a) {
        return a.has_bal == "True";
    });

    umap.append("g")
        .classed("bal-overlay", true)
        .selectAll("dot")
        .data(dataBAL)
        .enter()
        .append("circle")
        .attr("cx", function(d) { return umapX(+d.umap1); } )
        .attr("cy", function(d) { return umapY(+d.umap2); } )
        .attr("r", 2)
        .style("fill", "#173875")
        .style("opacity", 0.6);

    d3.select(".bal-ctrl").text("hide BALs");
};

const hideBALs = function() {
    umap.selectAll(".bal-overlay").remove();

    d3.select(".bal-ctrl").text("show BALs");
};

const drawUmap = function() {
    let plot = d3.select(".umap")
        .append("svg")
            .attr("width", umapWidth + umapMargin.left + umapMargin.right)
            .attr("height", umapHeight + umapMargin.top + umapMargin.bottom)
        .append("g")
            .attr("transform", "translate(" + umapMargin.left + ", " + umapMargin.top + ")");
    umap = plot;

    d3.select(".umap svg").append("svg:defs").append("svg:marker")
        .attr("id", "triangle")
        .attr("viewBox", "0 0 10 20")
        .attr("refX", 10)
        .attr("refY", 10)
        .attr("markerWidth", 6)
        .attr("markerHeight", 12)
        .attr("orient", "auto-start-reverse")
        .append("path")
            .attr("d", "M 0 0 L 10 10 L 0 20")
            .style("stroke", "black")
            .attr("stroke-width", 2)
            .attr("fill", "none");

    let padding = 0.5;
    let xBounds = d3.extent(data, function(d) { return +d.umap1 });
    umapX = d3.scaleLinear()
        .domain([xBounds[0] - padding, xBounds[1] + padding])
        .range([0, umapWidth]);
    plot.append("g")
        .attr("transform", "translate(0, " + umapHeight + ")")
        .call(d3.axisBottom(umapX).ticks(5));
    plot.append("text")
        .attr("text-anchor", "middle")
        .attr("x", umapWidth / 2)
        .attr("y", umapHeight + 34)
        .text("UMAP1");

    let yBounds = d3.extent(data, function(d) { return +d.umap2 });
    umapY = d3.scaleLinear()
        .domain([yBounds[0] - padding, yBounds[1] + padding])
        .range([umapHeight, 0]);
    plot.append("g")
        .call(d3.axisLeft(umapY).ticks(5));

    plot.append("text")
        .attr("text-anchor", "middle")
        .attr("transform", "rotate(-90)")
        .attr("y", -umapMargin.left + 20)
        .attr("x", -umapHeight / 2)
        .text("UMAP2");

    let sofaRange = d3.extent(data, function(d) { return +d.SOFA_score });
    let sofa = d3.scaleLinear()
        .domain(sofaRange)
        .range([umapHeight, 0]);
    distributionRanges["SOFA"] = sofaRange;
    distributionX["SOFA"] = sofa;

    colorUmap("clusters");
};

const addPatients = function() {
    let patientData = d3.filter(data, function(d) { return d.Patient_id !== "" });
    let patients = d3.group(patientData, function(d) { return d.Patient_id });
    let groups = d3.group(patients, function(d) { return d[1][0].Discharge_disposition });

    dischargeColor = d3.scaleOrdinal()
        .domain(DISCHARGE_COLORS.map(function(x) { return x[0] }))
        .range(DISCHARGE_COLORS.map(function(x) { return x[1] } ));

    d3.select(".patients")
        .selectAll("patient")
        .data(patients)
        .enter()
        .append("div")
            .classed("patient", true)
            .text(function(d) { return d[0] })
            .style("background", function(d) {
                return dischargeColor(d[1][0].Discharge_disposition)
            })
            .on("mouseenter", function(_, d) { showTrajectory(d) })
            .on("mouseleave", hideTrajectory)
            .on("click", function(_, d) {
                selectTrajectory(d);
            });

    d3.select(".legend")
        .selectAll("legend")
        .data(DISCHARGE_COLORS)
        .enter()
        .append("div")
            .classed("legend-item", true)
            .append("i")
                .classed("legend-marker", true)
                .style("background", function(d) {
                    return dischargeColor(d[0]).substr(0, 7);
                });
    d3.selectAll(".legend-item")
        .append("span")
            .text(function(d) { return d[0] + " (" + (groups.get(d[0]) || []).length + ")" });
};

const selectTrajectory = function(d) {
    selected = d;
    d3.select(".trajectory-hint").node().classList.remove("invisible");
    d3.select(".trends-ctrl").classed("disabled", false);
    d3.select(".trends-ctrl").on("click", function() {
        if (d3.select(".trends-ctrl").text() == "show trends") {
            showingTrends = true;
            showTrends();
        } else {
            showingTrends = false;
            showDistributions();
        }
    });
};

const showTrajectory = function(d) {
    let patientDisposition = d[1][0].Discharge_disposition;
    let color = dischargeColor(patientDisposition);
    let patientTitle = "Patient " + d[0] + " <small style='color:" + color + "'>(" + patientDisposition + ")</small>";
    d3.select(".trajectory").html("");
    d3.select(".trajectory-title")
        .classed("invisible", false)
        .select(".title")
            .html(patientTitle);
    d3.select(".trajectory-placeholder").classed("hidden", true);

    let width = d3.select(".first-row").node().offsetWidth;
    width -= d3.select(".umap").node().offsetWidth;

    let stays = d3.group(d[1], function(x) { return x.ICU_stay });
    let availableWidth = width - stays.size * margin.left - stays.size * margin.right;
    let perDayWidth = Math.floor(availableWidth / d[1].length);

    let plots = d3.select(".trajectory")
        .selectAll("stay")
        .data(stays)
        .enter()
        .append("svg")
            .attr("width", function(d) { return perDayWidth * d[1].length + margin.left + margin.right })
            .attr("height", height + margin.top + margin.bottom)
        .append("g")
            .attr("transform", "translate(" + margin.left + ", " + margin.top + ")");

    let xAxes = {};
    stays.forEach(function(stay, stayNum) {
        let xBounds = d3.extent(stay, function(d) { return +d.ICU_day });
        let x = d3.scaleLinear()
            .domain([xBounds[0], xBounds[1]])
            .range([0, perDayWidth * stay.length]);
        xAxes[stayNum] = x;
    });
    trajectoriesX = xAxes;

    plots.append("g")
        .attr("transform", "translate(0, " + height + ")")
        .each(function(d) {
            let xTicksNumber = Math.max(d[1].length - 1, 1);
            if (xTicksNumber > 100) {
                xTicksNumber = Math.round(xTicksNumber / 5);
            } else if (xTicksNumber > 50) {
                xTicksNumber = Math.round(xTicksNumber / 2);
            }
            const xAxisTicks = xAxes[d[0]].ticks(xTicksNumber)
                .filter(tick => Number.isInteger(tick));
            d3.axisBottom(xAxes[d[0]])
                .tickValues(xAxisTicks)
                .tickFormat(d3.format('d'))(d3.select(this));
        });
    
    plots.filter(function(_, i) { return i == 0 }).append("text")
        .attr("text-anchor", "middle")
        .attr("transform", "rotate(-90)")
        .attr("y", -margin.left + 16)
        .attr("x", -height / 2)
        .attr("font-size", 12)
        .text("SOFA score");

    plots.append("text")
        .attr("text-anchor", "middle")
        .attr("x", function(d) { return perDayWidth * d[1].length / 2 })
        .attr("y", height + 28)
        .attr("font-size", 12)
        .text("ICU day");

    let sofaMax = d3.max(d[1], function(d) { return +d.SOFA_score });
    let y = d3.scaleLinear()
        .domain([0, sofaMax])
        .range([height, 0]);
    trajectoryY = y;
    plots.append("g")
        .call(d3.axisLeft(y).ticks(5));

    plots.append("path")
        .attr("fill", "none")
        .attr("stroke", function(stay) {
            let color = dischargeColor(stay[1][0].Discharge_disposition);
            return color.substr(0, 7);
        })
        .attr("stroke-width", 1.9)
        .attr("d", function(stay) {
            if (stay[1].length == 1) {
                let x = stay[1][0];
                return d3.line()
                    .x(function(x) { return x[0] })
                    .y(function(x) { return y(+x[1].SOFA_score) })
                    ([[xAxes[stay[0]](+x.ICU_day) - 10, x], [xAxes[stay[0]](+x.ICU_day) + 10, x]]);
            }
            return d3.line()
                .x(function(x) { return xAxes[stay[0]](+x.ICU_day) })
                .y(function(x) { return y(+x.SOFA_score) })
                (stay[1]);
        });

    let bals = d3.filter(d[1], function(x) {
        return x.has_bal == "True";
    });
    bals = d3.group(bals, function(x) { return x.ICU_stay });

    plots.selectAll("bals")
        .data(function(stay) { return bals.get(stay[0]) || [] })
        .enter()
        .append("path")
            .attr("fill", "none")
            .attr("stroke", "#333333")
            .attr("stroke-width", 2)
            .attr("opacity", "0.8")
            .attr("d", function(bal) {
                let path = d3.path()
                path.moveTo(xAxes[bal.ICU_stay](+bal.ICU_day), y(0))
                path.lineTo(xAxes[bal.ICU_stay](+bal.ICU_day), y(sofaMax));
                return path.toString();
            });

    let episodes = d3.filter(d[1], function(x) {
        return x.Episode_is_cured !== "";
    });
    episodes = d3.group(episodes, function(x) { return x.ICU_stay });

    plots.selectAll("episodes")
        .data(function(stay) { return episodes.get(stay[0]) || [] })
        .enter()
        .append("rect")
            .attr("fill", function(epi) {
                if (epi.Episode_is_cured == "Cured") {
                    return "green";
                }
                if (epi.Episode_is_cured == "Indeterminate") {
                    return "yellow";
                }
                return "red";
            })
            .attr("x", function(epi) { return xAxes[epi.ICU_stay](+epi.ICU_day) })
            .attr("y", 0)
            .attr("height", height)
            .attr("width", function(epi) {
                let duration = epi.Episode_duration;
                if (duration === "") {
                    // NA duration
                    duration = 0; 
                } else {
                    duration = parseInt(epi.Episode_duration);
                }
                let x = xAxes[epi.ICU_stay];
                let maxDay = x.domain()[1];
                let day = +epi.ICU_day;
                day += duration;
                day = Math.min(day, maxDay);
                let width = x(day) - x(+epi.ICU_day);
                return width;
            })
            .attr("opacity", "0.2");

    plots.selectAll("episodes")
        .data(function(stay) { return episodes.get(stay[0]) || [] })
        .enter()
        .append("text")
            .attr("x", function(epi) { return xAxes[epi.ICU_stay](+epi.ICU_day) + 5 })
            .attr("y", height - 5)
            .text(function(epi) {
                let text;
                let etiology = epi.Episode_etiology;
                if (etiology == "Bacterial") {
                    etiology = "Bac.";
                } else if (etiology == "Bacterial/viral") {
                    etiology = "Bac/Vir.";
                } else if (etiology == "Culture-negative") {
                    etiology = "Cul. neg.";
                } else if (etiology == "Indeterminate") {
                    etiology = "Indeterm."
                }
                let duration = epi.Episode_duration;
                if (duration === "") {
                    // NA duration
                    duration = 0; 
                } else {
                    duration = parseInt(epi.Episode_duration);
                }
                text = etiology + " ";
                text += epi.Episode_category;
                if (duration > 0) {
                    text += " " + duration + "d";
                }
                text += ", " + epi.Episode_is_cured;
                return text;
            })
            .attr("font-size", 12);

    plots.selectAll("clusters")
        .data(function(stay) { return stay[1] })
        .enter()
        .append("rect")
            .attr("x", function(d) {
                let x = xAxes[d.ICU_stay];
                if (d.ICU_day - 1 < x.domain()[0]) {
                    return x(+d.ICU_day) - 10;
                }
                return x(+d.ICU_day - 0.5);
            })
            .attr("y", -16)
            .attr("width", function(d) {
                let x = xAxes[d.ICU_stay];
                let dayWidth = x(2) - x(1);
                if (+d.ICU_day + 1 > x.domain()[1]) {
                    if (stays.get(d.ICU_stay).length == 1) {
                        return 20;
                    }
                    return dayWidth / 2 + 9;
                }
                if (+d.ICU_day - 1 < x.domain()[0]) {
                    return dayWidth / 2 + 9;
                }
                return dayWidth - 1;
            })
            .attr("height", 12)
            .attr("fill", function(d) { return tab20[+d.clusters - 1] });
    plots.selectAll("clusters")
        .data(function(stay) { return stay[1] })
        .enter()
        .append("text")
            .attr("text-anchor", "middle")
            .attr("y", -6)
            .attr("x", function(d) { return xAxes[d.ICU_stay](+d.ICU_day) })
            .text(function(d) { return d.clusters })
            .attr("font-size", 12)
            .attr("font-weight", "bold");

    plots.append("text")
        // .attr("text-anchor", "end")
        .attr("y", height + 28)
        .attr("x", -15)
        .attr("font-size", 12)
        .text(function(stay) { return "ICU stay #" + stay[0] });

    umap.selectAll(".patient-overlay").remove();
    stays.forEach(function(stay) {
        umap.append("g")
            .classed("patient-overlay", true)
            .selectAll("selected")
            .data(stay)
            .enter()
            .append("circle")
                .attr("cx", function(d) { return umapX(+d.umap1); } )
                .attr("cy", function(d) { return umapY(+d.umap2); } )
                .attr("r", 2)
                .style("fill", "none")
                .style("stroke", "black")
                .style("stroke-width", 1.5);
        let arrows = [];
        for (let i = 0; i < stay.length - 1; i++) {
            let x1 = umapX(stay[i].umap1);
            let x2 = umapX(stay[i + 1].umap1);
            let y1 = umapY(stay[i].umap2);
            let y2 = umapY(stay[i + 1].umap2);
            let len = Math.sqrt((x2 - x1)**2 + (y2 - y1)**2);
            if (len < 10) {
                continue;
            }
            let w = 4 * (x2 - x1) / len;
            let arrowX = x2 - w;
            let arrowY = y2 - Math.sqrt(16 - w**2) * Math.sign(y2 - y1);
            arrows.push([x1, y1, arrowX, arrowY]);
        }
        umap.append("g")
            .classed("patient-overlay", true)
            .selectAll("arrows")
            .data(arrows)
            .enter()
            .append("line")
                .attr("x1", function(d) { return d[0] } )
                .attr("y1", function(d) { return d[1] } )
                .attr("x2", function(d) { return d[2] } )
                .attr("y2", function(d) { return d[3] } )
                .style("stroke", "black")
                .style("stroke-width", 1.2)
                .attr("marker-end", "url(#triangle)");
    });
    if (showingTrends) {
        showTrends(d);
    }

    d3.selectAll(".trajectory svg")
        .on("mousemove", onTrajectoryMouseMove)
        .on("mouseleave", onTrajectoryMouseLeave);

};

const hideTrajectory = function() {
    d3.select(".trajectory-title").classed("invisible", true);
    d3.select(".trajectory").html("");
    umap.selectAll(".patient-overlay").remove();
    d3.select(".trajectory-placeholder").classed("hidden", false);
    if (selected) {
        showTrajectory(selected);
    }
};

const onTrajectoryMouseMove = function(e, d) {
    let x = trajectoriesX[d[0]];
    let mouseX = d3.pointer(e, this)[0] - margin.left;

    let bestPoint = [];
    d[1].forEach(function(point) {
        let pointX = x(point.ICU_day);
        let dist = Math.abs(pointX - mouseX);
        if (bestPoint.length == 0) {
            bestPoint.push(dist);
            bestPoint.push(point);
        } else if (dist < bestPoint[0]) {
            bestPoint[0] = dist;
            bestPoint[1] = point;
        }
    });
    let point = bestPoint[1];
    if (currentPoint && currentPoint[""] == point[""]) {
        return;
    }
    let color = dischargeColor(point.Discharge_disposition);
    color = color.substr(0, 7);
    let svg = e.target;
    while (svg.nodeName != "svg" && svg.parentNode) {
        svg = svg.parentNode;
    }

    d3.select(svg).selectAll(".day-highlight").remove();
    d3.select(svg).append("circle")
        .classed("day-highlight", true)
        .attr("cx", x(+point.ICU_day) + margin.left)
        .attr("cy", trajectoryY(+point.SOFA_score) + margin.top)
        .attr("r", 5)
        .style("fill", color);

    umap.selectAll(".day-highlight").remove();
    umap.append("circle")
        .classed("day-highlight", true)
        .attr("cx", umapX(+point.umap1))
        .attr("cy", umapY(+point.umap2))
        .attr("r", 3)
        .style("fill", "#1e8920");
    umap.append("circle")
        .classed("day-highlight", true)
        .attr("cx", umapX(+point.umap1))
        .attr("cy", umapY(+point.umap2))
        .attr("r", 6)
        .style("fill", "none")
        .attr("stroke", "#1e8920")
        .attr("stroke-width", 2);
    umap.append("circle")
        .classed("day-highlight", true)
        .attr("cx", umapX(+point.umap1))
        .attr("cy", umapY(+point.umap2))
        .attr("r", 9)
        .style("fill", "none")
        .attr("stroke", "#1e8920")
        .attr("stroke-width", 2);
    
    showPointOnDistributions(point);
    currentPoint = point;
};

const onTrajectoryMouseLeave = function(e) {
    d3.selectAll(".distributions .point-overlay").remove();
    d3.select(e.target).selectAll(".day-highlight").remove();
    umap.selectAll(".day-highlight").remove();
    currentPoint = null;
};

const showDistribution = function(column) {
    let hist = d3.select(".distributions")
        .append("svg")
            .attr("width", histWidth + histMargin.left + histMargin.right)
            .attr("height", histHeight + histMargin.top + histMargin.bottom)
        .append("g")
            .attr("transform",
                  "translate(" + histMargin.left + "," + histMargin.top + ")");

    let xRange = d3.extent(data, function(d) {
        if (d[column] === "") {
            return NaN;
        }
        return +d[column];
    });
    let x = d3.scaleLinear()
        .domain(xRange)
        .range([0, histWidth]);
    distributionX[column] = x;
    distributionRanges[column] = xRange;
    hists[column] = hist;

    hist.append("g")
        .attr("transform", "translate(0, " + histHeight + ")")
        .attr("color", "#aaa")
        .call(d3.axisBottom(x).ticks(4).tickFormat(function(x) {
            if (+x >= 1000) {
                x = x / 1000;
                if (!Number.isInteger(x)) {
                    x = x.toFixed(1);
                }
                return x + "k";
            }
            return x;
        }));

    let histogram = d3.bin()
        .value(function(d) {
            if (d[column] === "") {
                return NaN;
            }
            return +d[column];
        })
        .domain(x.domain())
        .thresholds(x.ticks(20));

    let bins = histogram(data);

    let y = d3.scaleLinear()
        .range([histHeight, 0])
        .domain([0, d3.max(bins, function(d) { return d.length; })]);
    hist.append("g")
        .attr("color", "#ccc")
        .call(d3.axisLeft(y).ticks(0).tickSize(0));

    let binWidth = x(bins[1].x1) - x(bins[1].x0);
    hist.selectAll("rect")
        .data(bins)
        .enter()
        .append("rect")
            .attr("x", function(d) {
                // If second bin, special case
                let secondBin = bins[1];
                if (d.x0 == secondBin.x0) {
                    return (x(bins[0].x1) - x(bins[0].x0)) / 2 + 0.5;
                }
                if (d.x0 > xRange[0]) {
                    return x(d.x0) - binWidth / 2 + 0.5;
                }
                return x(d.x0) + 0.5;
            })
            .attr("transform", function(d) { return "translate(0, " + y(d.length) + ")"; })
            .attr("width", function(d) {
                if (d.x0 == xRange[0]) {
                    return (x(d.x1) - x(d.x0)) / 2;
                }
                let secondBin = bins[1];
                let firstBinWidth = (x(bins[0].x1) - x(bins[0].x0));
                if (d.x0 == secondBin.x0 && firstBinWidth < binWidth) {
                    return firstBinWidth / 2 + binWidth / 2;
                }
                let lastBin = bins[bins.length - 1];
                if (d.x0 == lastBin.x0) {
                    return binWidth / 2;
                }
                return binWidth;
            })
            .attr("height", function(d) { return histHeight - y(d.length); })
            .style("fill", "#aaa");

    hist.append("text")
        .classed("distr-title", true)
        .attr("text-anchor", "end")
        .attr("y", -5)
        .attr("x", histWidth)
        .attr("fill", "#999")
        .text(columnNames[column] || column)
        .on("click", function() { colorUmap(column) });
};

const showDistributions = function() {
    d3.selectAll(".distributions svg").remove();
    dataColumns.forEach(showDistribution);
    d3.select(".trends-ctrl").text("show trends");
};

const showTrend = function(p, column) {
    let hist = d3.select(".distributions")
        .append("svg")
            .attr("width", histWidth + histMargin.left + histMargin.right)
            .attr("height", histHeight + histMargin.top + histMargin.bottom)
        .append("g")
            .attr("transform",
                  "translate(" + histMargin.left + "," + histMargin.top + ")");

    let x = d3.scaleLinear()
        .domain([1, p[1].length])
        .range([0, histWidth]);
    distributionX[column] = x;
    distributionRanges[column] = [1, p[1].length];
    hists[column] = hist;

    let xTicksNumber = Math.max(p[1].length - 1, 1);
    if (xTicksNumber > 100) {
        xTicksNumber = Math.round(xTicksNumber / 5);
    } else if (xTicksNumber > 50) {
        xTicksNumber = Math.round(xTicksNumber / 2);
    }
    const xAxisTicks = x.ticks(8)
        .filter(tick => Number.isInteger(tick));

    hist.append("g")
        .attr("transform", "translate(0, " + histHeight + ")")
        .attr("color", "#aaa")
        .call(d3.axisBottom(x).tickValues(xAxisTicks).tickFormat(d3.format('d')));

    let yRange = d3.extent(p[1], function(d) { return +d[column] });

    let y = d3.scaleLinear()
        .range([histHeight, 0])
        .domain(yRange);
    hist.append("g")
        .attr("color", "#ccc")
        .call(d3.axisLeft(y).ticks(0).tickSize(0));

    hist.append("path")
        .attr("fill", "none")
        .attr("stroke", "#333")
        .attr("stroke-width", 1)
        .attr("d", function() {
            if (p[1].length == 1 && p[1][0][column] !== "") {
                let day = p[1][0];
                return d3.line()
                    .x(function(x) { return x })
                    .y(function(x) { return y(+x[column]) })
                    ([[x(1) - 10, day], [x(1) + 10, day]]);
            }
            let path = d3.path();
            let started = false;
            let lastValue, lastPathSize = 0;
            p[1].forEach(function(day, i) {
                if (day[column] === "") {
                    if (lastPathSize == 1) {
                        let x0 = x(i) - 2;
                        let x1 = x(i) + 2;
                        if (x0 > 0) {
                            path.moveTo(x0, lastValue);
                            path.lineTo(x1, lastValue);
                        } else {
                            path.lineTo(x1 + 1, lastValue);
                        }
                    }
                    started = false;
                    lastPathSize = 0;
                } else {
                    if (!started) {
                       path.moveTo(x(i + 1), y(+day[column]));
                       started = true;
                    } else {
                        path.lineTo(x(i + 1), y(+day[column]));
                    }
                    lastValue = y(+day[column]);
                    lastPathSize += 1;
                }
            });
            return path.toString();
        });

    hist.append("text")
        .classed("distr-title", true)
        .attr("text-anchor", "end")
        .attr("y", -5)
        .attr("x", histWidth)
        .attr("fill", "#999")
        .text(columnNames[column] || column)
        .on("click", function() { colorUmap(column) });
};

const showTrends = function(d) {
    d3.selectAll(".distributions svg").remove();
    if (d === undefined) {
        d = selected;
    }
    dataColumns.forEach(function(column) { showTrend(d, column) });
    d3.select(".trends-ctrl").text("show distributions");
};

const showPointOnDistributions = function(point) {
    d3.selectAll(".distributions .point-overlay").remove();
    dataColumns.forEach(function(column) {
        let hist = hists[column];
        let x = distributionX[column];
        let isMissing = point[column] === "";
        if (!isMissing || showTrends) {
            let dayX = x(+point[column]);
            if (showingTrends) {
                let i = 0;
                for (; i < selected[1].length; i++) {
                    if (selected[1][i][""] === point[""]) {
                        break;
                    }
                }
                dayX = x(i + 1);
            }
            hist.append("line")
                .classed("point-overlay", true)
                .attr("x1", dayX)
                .attr("y1", histHeight)
                .attr("x2", dayX)
                .attr("y2", 0)
                .attr("stroke", "black")
                .attr("stroke-width", 1.5);
        }
        if (isMissing) {
            hist.append("text")
                .classed("point-overlay", true)
                .attr("text-anchor", "start")
                .attr("x", 0)
                .attr("y", -5)
                .attr("fill", "#e00a06")
                .attr("font-size", 12)
                .text("NA");
        } else {
            hist.append("text")
                .classed("point-overlay", true)
                .attr("text-anchor", "start")
                .attr("x", 0)
                .attr("y", -5)
                .attr("font-size", 12)
                .text((+point[column]).toLocaleString(undefined, { maximumFractionDigits: 2 }));
        }
    });
};

const onLoad = function(d) {
    d = pako.inflate(d, {to: "string"});
    d = d3.csvParse(d);
    data = d.sort(function(a, b) {
        let result = a.Patient_id - b.Patient_id;
        if (result != 0) {
            return result;
        }
        result = a.ICU_stay - b.ICU_stay;
        if (result != 0) {
            return result;
        }
        return a.ICU_day - b.ICU_day;
    });

    drawUmap();
    addPatients();
    showDistributions();
    d3.select(".color-reset").on("click", function() { colorUmap("clusters") });
    d3.select(".bal-ctrl").on("click", function() {
        if (d3.select(".bal-ctrl").text() == "show BALs") {
            showBALs();
        } else {
            hideBALs();
        }
    });

    d3.select(".loader").classed("hidden", true);
    d3.select(".first-row").classed("hidden", false);
    d3.select(".patient-legend").classed("hidden", false);
    d3.select(".patients").classed("hidden", false);
};
