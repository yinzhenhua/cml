package gocalendar

import "fmt"

//isLearYear 判断年份是否是润年
func isLeapYear(year int) bool {
	return year%400 == 0 || year%4 == 0 && year%100 != 0
}

//getMonthDays 获取月份对应的月份天数
func getMonthDays(year, month int) int {
	days := 30
	switch {
	case month == 1 || month == 3 || month == 5 || month == 7 || month == 8 || month == 10 || month == 12:
		days = 31
	case month == 2:
		if isLeapYear(year) {
			days = 29
		} else {
			days = 28
		}
	default:
		days = 30
	}
	return days
}

//getYearDays 获取年份对应的天数
func getYearDays(year int) int {
	if isLeapYear(year) {
		return 366
	} else {
		return 365
	}
}

//getTotalDays 获取所有的天数
//以1970年1月1日为参考依据
func getTotalDays(year, month int) int {
	days := 0
	for i := 1900; i < year; i++ {
		days += getYearDays(i)
	}
	for i := 1; i < month; i++ {
		days += getMonthDays(year, i)
	}
	return days
}

//getWeekDay 根据年份和月份获取对应的星期
func getWeekDay(year, month int) int {
	days := getTotalDays(year, month) + 1
	weekDay := days % 7
	return weekDay
}

//PrintCalendar 打印出
func PrintCalendar(year, month int) {
	weekday := getWeekDay(year, month)
	fmt.Println("------------------------------")
	fmt.Printf("%s\t%s\t%s\t%s\t%s\t%s\t%s\n", "日", "一", "二", "三", "四", "五", "六")
	fmt.Println("------------------------------")
	//打印空格
	for i := 0; i < weekday; i++ {
		fmt.Print("\t")
	}
	monthDay := getMonthDays(year, month)
	for i := 1; i < monthDay+1; i++ {
		fmt.Printf("%d\t", i)
		if (i+weekday)%7 == 0 {
			fmt.Println()
		}
	}
}
