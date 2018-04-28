package main

import (
	"goml/core"
	"path"
	"fmt"
)

//readData 读取训练数据
func readData() (core.Matrix, core.Matrix) {
	filePath := path.Join("data", "aqi1.csv")
	data := core.ReadFromCSV(filePath, -1)
	rows, cols := data.Shape()
	trainData := data.MatrixArea(0, rows, 0, cols-1)
	//对数据进行归一化处理
	mean := trainData.Mean()
	standard := trainData.Standard()
	trainData, err := trainData.MinusValue(mean).DivideValue(standard)
	if err != nil {
		panic(err.Error())
	}
	trainLabel := data.MatrixArea(0, rows, cols-1, cols)
	return trainData, trainLabel
}

//readTestData 读取测试数据
func readTestData() (core.Matrix, core.Matrix) {
	filePath := path.Join("data", "aqi_test.csv")
	data := core.ReadFromCSV(filePath, -1)
	rows, cols := data.Shape()
	testData := data.MatrixArea(0, rows, 0, cols-1)
	//对数据进行归一化处理
	mean := testData.Mean()
	standard := testData.Standard()
	testData, err := testData.MinusValue(mean).DivideValue(standard)
	if err != nil {
		panic(err.Error())
	}
	testLabel := data.MatrixArea(0, rows, cols-1, cols)
	return testData, testLabel
}
func main() {
	data, label := readData()
	linearRegression := core.LinearRegression{}
	model := linearRegression.Fit(data, label)
	testData, testLabel := readTestData()
	result := model.Predicate(testData)
	source := model.Score(testLabel, result)
	fmt.Println(source)
}
