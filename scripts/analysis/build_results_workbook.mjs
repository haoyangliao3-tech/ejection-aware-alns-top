import fs from "node:fs/promises";
import path from "node:path";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";


const inputPath = path.resolve(process.argv[2] ?? "data/results/normalized_results.json");
const outputPath = path.resolve(process.argv[3] ?? "data/results/Detailed_Instance_Results.xlsx");
const previewDir = path.resolve(process.argv[4] ?? "tmp/workbook_previews");
const data = JSON.parse(await fs.readFile(inputPath, "utf8"));
await fs.mkdir(path.dirname(outputPath), { recursive: true });
await fs.mkdir(previewDir, { recursive: true });

const workbook = Workbook.create();
const C = {
  navy: "#17365D", teal: "#0F6B78", tealLight: "#D9EEF2",
  blueLight: "#EAF1F8", border: "#CBD2D9", white: "#FFFFFF",
  green: "#E2F0D9", red: "#FCE4D6",
};

function col(index) {
  let value = index + 1;
  let result = "";
  while (value > 0) {
    result = String.fromCharCode(65 + ((value - 1) % 26)) + result;
    value = Math.floor((value - 1) / 26);
  }
  return result;
}

function rows(records, headers) {
  return records.map((record) => headers.map((header) => record[header] ?? null));
}

function formatColumns(sheet, headers, lastRow) {
  headers.forEach((header, index) => {
    const letter = col(index);
    const range = sheet.getRange(`${letter}2:${letter}${lastRow}`);
    if (/P_Value|\bp\b/i.test(header)) range.format.numberFormat = "0.00E+00";
    else if (/Gap_Percent|Rate|Share|Biserial/i.test(header)) range.format.numberFormat = "0.000000";
    else if (/Runtime|Seconds|Distance|Reward|BKS$|Gain/i.test(header)) range.format.numberFormat = "0.000";
    else if (/Seed|Iterations|Customers|Calls|Commits|Hits|Instances|Runs|Workers|Bytes/i.test(header)) range.format.numberFormat = "0";

    let width = 15;
    if (/Source_Record|Expected_Relative_Path/.test(header)) width = 29;
    else if (/Instance/.test(header)) width = 22;
    else if (/Algorithm|Config|Status|Role|Ranking|Protocol|Benchmark/.test(header)) width = 23;
    else if (/SHA256/.test(header)) width = 36;
    else if (/Gap|Runtime|Seconds|Distance|Iterations/.test(header)) width = 19;
    sheet.getRange(`${letter}1:${letter}${lastRow}`).format.columnWidth = width;
  });
}

function addSheet(name, records, freezeColumns = 3) {
  if (!records.length) throw new Error(`Empty worksheet: ${name}`);
  const sheet = workbook.worksheets.add(name);
  sheet.showGridLines = false;
  const baseHeaders = Object.keys(records[0]);
  const hasGap = baseHeaders.includes("BKS_Gap_Percent") && baseHeaders.includes("BKS") && baseHeaders.includes("Reward");
  const headers = hasGap ? [...baseHeaders, "Calculated_Gap_Percent", "Gap_Check"] : baseHeaders;
  const lastRow = records.length + 1;
  const lastCol = col(headers.length - 1);
  sheet.getRange(`A1:${lastCol}${lastRow}`).values = [
    headers,
    ...rows(records, baseHeaders).map((row) => hasGap ? [...row, null, null] : row),
  ];
  if (hasGap) {
    const bks = col(headers.indexOf("BKS"));
    const reward = col(headers.indexOf("Reward"));
    const source = col(headers.indexOf("BKS_Gap_Percent"));
    const calculated = col(headers.indexOf("Calculated_Gap_Percent"));
    const check = col(headers.indexOf("Gap_Check"));
    sheet.getRange(`${calculated}2`).formulas = [[`=IFERROR(100*(${bks}2-${reward}2)/${bks}2,"")`]];
    sheet.getRange(`${calculated}2:${calculated}${lastRow}`).fillDown();
    sheet.getRange(`${check}2`).formulas = [[`=IF(ABS(${source}2-${calculated}2)<=0.000001,"OK","CHECK")`]];
    sheet.getRange(`${check}2:${check}${lastRow}`).fillDown();
    sheet.getRange(`${check}2:${check}${lastRow}`).conditionalFormats.add("containsText", {
      text: "CHECK", format: { fill: C.red, font: { bold: true, color: "#9C0006" } },
    });
    sheet.getRange(`${check}2:${check}${lastRow}`).conditionalFormats.add("containsText", {
      text: "OK", format: { fill: C.green, font: { color: "#006100" } },
    });
  }
  const populated = sheet.getRange(`A1:${lastCol}${lastRow}`);
  populated.format.font = { name: "Aptos", size: 9, color: "#1F2933" };
  populated.format.borders = { insideHorizontal: { style: "thin", color: C.border } };
  sheet.getRange(`A1:${lastCol}1`).format = {
    fill: C.teal, font: { name: "Aptos", size: 9, bold: true, color: C.white },
    wrapText: true, verticalAlignment: "center", horizontalAlignment: "center", rowHeight: 36,
  };
  formatColumns(sheet, headers, lastRow);
  sheet.freezePanes.freezeRows(1);
  sheet.freezePanes.freezeColumns(freezeColumns);
  const tableName = `${name.replace(/[^A-Za-z0-9]/g, "")}Table`;
  sheet.tables.add(`A1:${lastCol}${lastRow}`, true, tableName).style = "TableStyleMedium2";
  return { sheet, lastRow, lastCol };
}

const readme = workbook.worksheets.add("README");
readme.showGridLines = false;
readme.mergeCells("A1:H2");
readme.getRange("A1").values = [["Corrected Detailed Computational Results"]];
readme.getRange("A1:H2").format = {
  fill: C.navy, font: { name: "Aptos Display", size: 20, bold: true, color: C.white },
  verticalAlignment: "center",
};
readme.mergeCells("A4:H5");
readme.getRange("A4").values = [["Unified workers=2 experiments on the Dang and Chao canonical TOP benchmarks"]];
readme.getRange("A4:H5").format = {
  fill: C.blueLight, font: { name: "Aptos", size: 12, bold: true, color: C.navy },
  wrapText: true, verticalAlignment: "center",
};
const notes = [
  ["Purpose", data.metadata.description],
  ["Benchmarks", "Dang: 82 instances; Chao Sets 4-7: 157 instances."],
  ["Concurrency", data.metadata.worker_definition],
  ["BKS gap", data.metadata.gap_definition],
  ["Fixed iteration", "Ejection ON/OFF, GRASP, ILS, and VNS; four seeds and 2,500 iterations."],
  ["Symmetric wall clock", "Ejection ON and OFF rerun under the same instance-specific hard-wall budgets."],
  ["Runtime baselines", "GRASP, ILS, VNS, and PyVRP rerun under those same instance-specific budgets."],
  ["Sensitivity", "Seven configurations on nine instances per benchmark, two seeds, and 5,000 iterations."],
  ["Mechanism study", "Seven configurations on nine instances per benchmark, two seeds, and 2,500 iterations."],
  ["No pooled claim", "Dang and Chao summaries are reported separately; the workbook does not pool their gaps."],
  ["Gap verification", "Calculated gap columns use workbook formulas; Gap_Check should read OK."],
];
readme.getRange(`A7:B${6 + notes.length}`).values = notes;
readme.getRange(`A7:A${6 + notes.length}`).format = { fill: C.tealLight, font: { name: "Aptos", size: 10, bold: true, color: C.navy } };
readme.getRange(`B7:B${6 + notes.length}`).format = { font: { name: "Aptos", size: 10 }, wrapText: true };
readme.getRange(`A7:B${6 + notes.length}`).format.borders = { insideHorizontal: { style: "thin", color: C.border }, outside: { style: "thin", color: C.border } };
readme.getRange("A1:A28").format.columnWidth = 24;
readme.getRange("B1:B28").format.columnWidth = 82;
readme.freezePanes.freezeRows(2);

addSheet("Benchmark_BKS", data.benchmark_bks, 2);
addSheet("Fixed_Iteration", data.fixed_iteration);
addSheet("Symmetric_Wall_Clock", data.symmetric_wall_clock);
addSheet("Runtime_Baselines", data.runtime_budget_baselines);
addSheet("Runtime_Budgets", data.runtime_budgets, 2);
addSheet("Sensitivity", data.sensitivity);
addSheet("Focused_Mechanism", data.focused_mechanism);
addSheet("QA_Summary", data.qa_summary, 3);
addSheet("Paired_Statistics", data.reported_statistics, 4);
addSheet("Component_Statistics", data.component_statistics, 3);

const formulaErrors = await workbook.inspect({
  kind: "match", searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
  options: { useRegex: true, maxResults: 200 }, summary: "formula error scan",
});
await fs.writeFile(path.join(previewDir, "formula_error_scan.ndjson"), formulaErrors.ndjson, "utf8");

for (const [sheetName, range] of [
  ["README", "A1:H19"], ["Fixed_Iteration", "A1:R16"],
  ["Symmetric_Wall_Clock", "A1:X16"], ["Runtime_Baselines", "A1:W16"],
  ["QA_Summary", "A1:M24"], ["Paired_Statistics", "A1:N20"],
]) {
  const image = await workbook.render({ sheetName, range, scale: 1, format: "png" });
  await fs.writeFile(path.join(previewDir, `${sheetName}.png`), new Uint8Array(await image.arrayBuffer()));
}

const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(outputPath);
console.log(JSON.stringify({ outputPath, sheets: 11, metadata: data.metadata }, null, 2));
