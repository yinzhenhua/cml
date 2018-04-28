package core

import (
	"math"
	"os"
	"encoding/csv"
	"io"
	"strconv"
)

//Matrix 矩阵
type Matrix [][]float64

//Shape 获取二维矩阵的大小
func (m Matrix) Shape() (int, int) {
	rows := len(m)
	cols := len(m[0])
	return rows, cols
}

//Sigmoid 非线性函数
func (m Matrix) Sigmoid() Matrix {
	length := len(m)
	var result Matrix = make([][]float64, length)
	for i := 0; i < length; i++ {
		subLength := len(m[i])
		result[i] = make([]float64, subLength)
		for j := 0; j < subLength; j++ {
			result[i][j] = 1.0 / (1.0 + math.Exp(m[i][j]))
		}
	}
	return result
}

//Transform 矩阵转置
func (m Matrix) Transform() Matrix {
	rows, cols := m.Shape()
	var result Matrix = make([][]float64, cols)
	for i := 0; i < cols; i++ {
		result[i] = make([]float64, rows)
		for j := 0; j < rows; j++ {
			result[i][j] = m[j][i]
		}
	}
	return result
}

//Sum 求矩阵中所有元素的和
func (m Matrix) Sum() float64 {
	sum := 0.0
	rows, cols := m.Shape()
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			sum += m[i][j]
		}
	}
	return sum
}

//Mean 求矩阵中所有元素的均值
func (m Matrix) Mean() float64 {
	row, cols := m.Shape()
	sum := m.Sum()
	return sum / float64(row*cols)
}

//Standard 求矩阵元素的标准差
func (m Matrix) Standard() float64 {
	delta := 0.0
	rows, cols := m.Shape()
	mean := m.Mean()
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			delta += math.Pow(m[i][j]-mean, 2)
		}
	}
	return math.Sqrt(delta / float64(rows*cols))
}

//Multiply 矩阵与单个数值相乘，矩阵中的每个元素均与该数值相乘
func (m Matrix) Multiply(num float64) Matrix {
	rows, cols := m.Shape()
	var result Matrix = make([][]float64, rows)
	for i := 0; i < rows; i++ {
		result[i] = make([]float64, cols)
		for j := 0; j < cols; j++ {
			result[i][j] = m[i][j] * num
		}
	}
	return result
}

// MultiplyMatrix 矩阵相同的位置相乘
func (m Matrix) MultiplyMatrix(other Matrix) Matrix {
	rows, cols := m.Shape()
	var result Matrix = make([][]float64, rows)
	for i := 0; i < rows; i++ {
		result[i] = make([]float64, cols)
		for j := 0; j < cols; j++ {
			result[i][j] = m[i][j] * other[i][j]
		}
	}
	return result
}

//Dot 矩阵乘法
func (m Matrix) Dot(other Matrix) Matrix {
	rows, cols := m.Shape()
	_, cols1 := other.Shape()
	var result Matrix = make([][]float64, rows)
	for i := 0; i < rows; i++ {
		result[i] = make([]float64, cols1)
		for j := 0; j < cols; j++ {
			for z := 0; z < cols1; z++ {
				result[i][z] += m[i][j] * other[j][z]
			}
		}
	}
	return result
}

//Range 获取矩阵区域数值
func (m Matrix) MatrixArea(rowStart, rowEnd, colStart, colEnd int) Matrix {
	var result Matrix = make([][]float64, rowEnd-rowStart)
	rowIndex, colIndex := 0, 0
	for i := rowStart; i < rowEnd; i++ {
		result[rowIndex] = make([]float64, colEnd-colStart)
		colIndex = 0
		for j := colStart; j < colEnd; j++ {
			result[rowIndex][colIndex] = m[i][j]
			colIndex++
		}
		rowIndex++
	}
	return result
}

//AddValue 矩阵与元素相加，将矩阵中每个元素与指定的值相加，并返回新的矩阵
func (m Matrix) AddValue(value float64) Matrix {
	rows, cols := m.Shape()
	var result Matrix = make([][]float64, rows)
	for i := 0; i < rows; i++ {
		result[i] = make([]float64, cols)
		for j := 0; j < cols; j++ {
			result[i][j] = m[i][j] + value
		}
	}
	return result
}

//AddMatrix 矩阵与矩阵相加
func (m Matrix) AddMatrix(other Matrix) Matrix {
	rows, cols := m.Shape()
	var result Matrix = make([][]float64, rows)
	for i := 0; i < rows; i++ {
		result[i] = make([]float64, cols)
		for j := 0; j < cols; j++ {
			result[i][j] = m[i][j] + other[i][j]
		}
	}
	return result
}

//ReadFromCSV 从CSV文件中读取信息，并返回对应的矩阵
func ReadFromCSV(csvFile string, headLine int) Matrix {
	file, err := os.Open(csvFile)
	if err != nil {
		panic(err.Error())
	}
	defer file.Close()
	var result Matrix = make([][]float64, 0)
	readeLines := 0
	csvReader := csv.NewReader(file)
	for {
		record, err := csvReader.Read()
		if err != nil {
			if err == io.EOF {
				break
			} else {
				panic(err.Error())
			}
		}
		readeLines++
		if readeLines < headLine {
			continue
		}
		//从Record中提取出每个记录并存储到矩阵中
		matrixItem := make([]float64, len(record))
		for index, item := range record {
			floatValue, err := strconv.ParseFloat(item, 32)
			if err != nil {
				panic(err.Error())
			}
			matrixItem[index] = floatValue
		}
		result = append(result, matrixItem)
	}
	return result
}
