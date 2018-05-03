package main

import (
	"goml/core"
	"path"
	"fmt"
	"goml/gocalendar"
)

//readData 读取训练数据
func readData(head int) (core.Matrix, core.Matrix) {
	filePath := path.Join("data", "adult.csv")
	data := core.ReadFromCSV(filePath, head)
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
func readTestData(head int) (core.Matrix, core.Matrix) {
	filePath := path.Join("data", "adult_test.csv")
	data := core.ReadFromCSV(filePath, head)
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
	// 逻辑回归测试
	data, label := readData(1)
	regression := core.LogisticRegression{}
	model := regression.Fit(data, label)
	testData, testLabel := readTestData(1)
	result := model.Predicate(testData)
	source := model.Score(testLabel, result)
	fmt.Printf("逻辑回归准确率为：%f\n",source)

	// 线性回归测试

	// 打印日历
	gocalendar.PrintCalendar(2018,5)

}
