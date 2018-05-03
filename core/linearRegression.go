package core

import (
	"math"
)

//LinearModel 线性模型
type LinearModel struct {
	Model Matrix
}

//LogisticModel 逻辑模型
type LogisticModel struct {
	Model Matrix
}

//LinearRegression 线性回归
type LinearRegression struct{}

//LogisticRegression 逻辑回归
type LogisticRegression struct {
}

/*
 * 使用线性回归模型对线性数据进行建模
 *  trainData:训练数据
 *  trainLabel:训练数据对应的标签
 *默认的学习率为0.01,迭代运算的次数10000次
 */
func (linear LinearRegression) Fit(trainData, trainLabel Matrix) LinearModel {
	theta := linear.fit(trainData, trainLabel, 0.01, 50000)
	rows, cols := theta.Shape()
	var model Matrix = make([][]float64, rows)
	for i := 0; i < rows; i++ {
		model[i] = make([]float64, cols)
		{
			for j := 0; j < cols; j++ {
				model[i][j] = theta[i][j]
			}
		}
	}
	return LinearModel{Model: model}
}

//Fit 逻辑回归的Fit
func (logistic LogisticRegression) Fit(trainData, trainLabel Matrix) LogisticModel {
	theta := logistic.fit(trainData, trainLabel, 0.05, 10000)
	rows, cols := theta.Shape()
	var model Matrix = make([][]float64, rows)
	for i := 0; i < rows; i++ {
		model[i] = make([]float64, cols)
		{
			for j := 0; j < cols; j++ {
				model[i][j] = theta[i][j]
			}
		}
	}
	return LogisticModel{Model: model}
}

func (logistic LogisticRegression) fit(trainData, trainLabel Matrix, alpha float64, iteratorSteps int) Matrix {
	rows, cols := trainData.Shape()
	// 在所有数据前面加一列
	data := Ones(rows, cols+1)
	MatrixCopy(data, 1, trainData, 0)
	//构建theta
	theta := Zeros(cols+1, 1)
	return logisticGradientDescending(data, trainLabel, theta, alpha, iteratorSteps)
}

//logisticGradientDescending 逻辑回归梯度下降算法
func logisticGradientDescending(trainData, trainLabel, theta Matrix, alpha float64, iteratorSteps int) Matrix {
	m := len(trainData)
	for i := 0; i < iteratorSteps; i++ {
		h := trainData.Dot(theta).Sigmoid()
		tmp, err := h.MinusMatrix(trainLabel).Multiply(alpha).DivideValue(float64(m))
		if err != nil {
			panic(err.Error())
		}
		theta = theta.MinusMatrix(tmp)
		upgradeAlpha(alpha, int64(m), 0.001)
	}
	return theta
}

// Predicate 使用模型进行预测
func (model LinearModel) Predicate(predicateData Matrix) Matrix {
	return predicateData.Dot(model.Model)
}

//Predicate 使用逻辑回归模型进行预测
func (model LogisticModel) Predicate(predicateData Matrix) Matrix {
	result := predicateData.Dot(model.Model).Sigmoid()
	rows, cols := result.Shape()
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			result[i][j] = float64(getLogisticValue(result[i][j]))
		}
	}
	return result
}

func getLogisticValue(value float64) int {
	switch {
	case value > 0.5:
		return 1
	default:
		return 0
	}
}

//Score 模型得分
func (model LinearModel) Score(testLabel, predicateResult Matrix) float64 {
	rows, _ := testLabel.Shape()
	return math.Sqrt(predicateResult.MinusMatrix(testLabel).Power().Sum()) / float64(rows)
}

//Score 对逻辑回归的结果进行打分
func (model LogisticModel) Score(testLabel, predicateResult Matrix) float64 {
	rows, cols := testLabel.Shape()
	var sum int = 0
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			if testLabel[i][j] == predicateResult[i][j] {
				sum += 1
			}
		}
	}
	return float64(sum) / float64(rows)
}

//fit 使用线性模型对现有数据进行线性回归
func (linear LinearRegression) fit(trainData, trainLabel Matrix, alpha float64, iteratorSteps int) Matrix {
	rows, cols := trainData.Shape()
	// 在所有数据前面加一列
	data := Ones(rows, cols+1)
	MatrixCopy(data, 1, trainData, 0)
	//构建theta
	theta := Zeros(cols+1, 1)
	return gradientDescending(data, trainLabel, theta, alpha, iteratorSteps)
}

//gradientDescending 梯度下降算法
func gradientDescending(trainData, trainLabel, theta Matrix, alpha float64, iteratorSteps int) Matrix {
	m, _ := trainData.Shape()
	var numbers int64 = int64(len(trainLabel))
	xt := trainData.Transform()
	for i := 0; i < iteratorSteps; i++ {
		h := trainData.Dot(theta)
		g := xt.Dot(h.MinusMatrix(trainLabel)).Multiply(alpha)
		upgradeAlpha(alpha, numbers, 0.001)
		mi, err := g.DivideValue(float64(m))
		if err != nil {
			panic(err.Error())
		}
		theta = theta.MinusMatrix(mi)
	}
	return theta
}

//upgradeAlpha 更新alpha
//参考退火公式 alpha = alpha/(1+d*t)
//其中d = 0.001
func upgradeAlpha(alpha float64, numbers int64, alphaDesc float64) float64 {
	return alpha / (1 + alphaDesc*float64(numbers))
}
