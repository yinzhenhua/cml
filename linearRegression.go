package main

import (
	"fmt"
	"goml/core"
)


func main() {
	var m core.Matrix = [][]float64{{1, 2, 3}, {1, 2, 3}, {3, 4, 2}}
	m1 := m.AddValue(5.0)
	fmt.Println(m1)
}
