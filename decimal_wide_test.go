package duckdb

import (
	"context"
	"database/sql/driver"
	"fmt"
	"math/big"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/duckdb/duckdb-go/v2/mapping"
)

// Wide decimals are DECIMAL types with more than 38 digits.
// DuckDB stores their unscaled value as BIGNUM.

// wideDecimalText returns a decimal literal with width digits, scale of which are fractional digits.
func wideDecimalText(width, scale int, negative bool) string {
	digits := strings.Repeat("1234567890", width/10+1)[:width]
	text := digits
	if scale > 0 {
		text = digits[:width-scale] + "." + digits[width-scale:]
	}
	if negative {
		text = "-" + text
	}
	return text
}

// wideDecimal returns the Decimal matching wideDecimalText.
func wideDecimal(t *testing.T, width, scale int, negative bool) Decimal {
	text := strings.ReplaceAll(wideDecimalText(width, scale, negative), ".", "")
	unscaled, ok := new(big.Int).SetString(text, 10)
	require.True(t, ok)
	return Decimal{Width: uint8(width), Scale: uint8(scale), Value: unscaled}
}

type wideDecimalSUDF struct {
	info TypeInfo
}

func (u *wideDecimalSUDF) Config() ScalarFuncConfig {
	return ScalarFuncConfig{InputTypeInfos: []TypeInfo{u.info}, ResultTypeInfo: u.info}
}

func (*wideDecimalSUDF) Executor() ScalarFuncExecutor {
	return ScalarFuncExecutor{RowExecutor: func(values []driver.Value) (any, error) {
		if values[0] == nil {
			return nil, nil
		}
		d, ok := values[0].(Decimal)
		if !ok {
			return nil, fmt.Errorf("expected Decimal, got %T", values[0])
		}
		d.Value = new(big.Int).Neg(d.Value)
		return d, nil
	}}
}

func TestWideDecimalScan(t *testing.T) {
	db := openDbWrapper(t, ``)
	defer closeDbWrapper(t, db)

	tests := []struct {
		width, scale int
	}{
		{width: 39, scale: 0},
		{width: 39, scale: 2},
		{width: 39, scale: 39},
		{width: 76, scale: 10},
		{width: 255, scale: 0},
		{width: 255, scale: 100},
		{width: 255, scale: 255},
	}
	for _, tt := range tests {
		for _, negative := range []bool{false, true} {
			name := fmt.Sprintf("DECIMAL(%d,%d) negative=%t", tt.width, tt.scale, negative)
			t.Run(name, func(t *testing.T) {
				text := wideDecimalText(tt.width, tt.scale, negative)
				query := fmt.Sprintf(`SELECT '%s'::DECIMAL(%d,%d)`, text, tt.width, tt.scale)
				want := wideDecimal(t, tt.width, tt.scale, negative)

				var got Decimal
				require.NoError(t, db.QueryRow(query).Scan(&got))
				compareDecimal(t, want, got)

				// Scanning into any yields a Decimal, never a float64.
				var v any
				require.NoError(t, db.QueryRow(query).Scan(&v))
				require.IsType(t, Decimal{}, v)
				compareDecimal(t, want, v.(Decimal))

				// The exact text is available through a cast.
				var str string
				require.NoError(t, db.QueryRow(query+`::VARCHAR`).Scan(&str))
				require.Equal(t, text, str)
			})
		}
	}

	t.Run("zero and small values", func(t *testing.T) {
		rows, err := db.Query(`SELECT d::DECIMAL(50,3) FROM (VALUES (0), (1), (-1), (0.001), (-0.001), (NULL)) t(d)`)
		require.NoError(t, err)
		defer closeRowsWrapper(t, rows)

		want := []*big.Int{big.NewInt(0), big.NewInt(1000), big.NewInt(-1000), big.NewInt(1), big.NewInt(-1), nil}
		i := 0
		for rows.Next() {
			var got *Decimal
			require.NoError(t, rows.Scan(&got))
			if want[i] == nil {
				require.Nil(t, got)
			} else {
				compareDecimal(t, Decimal{Width: 50, Scale: 3, Value: want[i]}, *got)
			}
			i++
		}
		require.NoError(t, rows.Err())
		require.Equal(t, len(want), i)
	})

	t.Run("default session arithmetic", func(t *testing.T) {
		// Adding two DECIMAL(38,2) values yields a DECIMAL(39,2).
		query := `SELECT typeof(a + b), a + b FROM (SELECT
			999999999999999999999999999999999999.99::DECIMAL(38,2) AS a,
			999999999999999999999999999999999999.99::DECIMAL(38,2) AS b)`
		var typ string
		var got Decimal
		require.NoError(t, db.QueryRow(query).Scan(&typ, &got))
		require.Equal(t, "DECIMAL(39,2)", typ)
		unscaled, ok := new(big.Int).SetString("199999999999999999999999999999999999998", 10)
		require.True(t, ok)
		compareDecimal(t, Decimal{Width: 39, Scale: 2, Value: unscaled}, got)
		require.Equal(t, "1999999999999999999999999999999999999.98", got.String())
	})

	t.Run("many rows", func(t *testing.T) {
		// Spans several data chunks, with inlined and non-inlined BIGNUM payloads.
		rows, err := db.Query(`SELECT i, (i::DECIMAL(60,0) * 100000000000000000000000000000000000000 - 5)::DECIMAL(60,5)
			FROM range(5000) t(i)`)
		require.NoError(t, err)
		defer closeRowsWrapper(t, rows)

		factor, ok := new(big.Int).SetString("100000000000000000000000000000000000000", 10)
		require.True(t, ok)
		count := 0
		for rows.Next() {
			var i int64
			var got Decimal
			require.NoError(t, rows.Scan(&i, &got))
			want := new(big.Int).Mul(big.NewInt(i), factor)
			want.Sub(want, big.NewInt(5))
			want.Mul(want, big.NewInt(100000))
			compareDecimal(t, Decimal{Width: 60, Scale: 5, Value: want}, got)
			count++
		}
		require.NoError(t, rows.Err())
		require.Equal(t, 5000, count)
	})

	t.Run("column type", func(t *testing.T) {
		rows, err := db.Query(`SELECT 1::DECIMAL(76,10) AS d`)
		require.NoError(t, err)
		defer closeRowsWrapper(t, rows)
		types, err := rows.ColumnTypes()
		require.NoError(t, err)
		require.Equal(t, "DECIMAL(76,10)", types[0].DatabaseTypeName())
		require.Equal(t, reflectTypeDecimal, types[0].ScanType())
	})
}

func TestWideDecimalNested(t *testing.T) {
	db := openDbWrapper(t, ``)
	defer closeDbWrapper(t, db)

	pos := wideDecimal(t, 76, 10, false)
	neg := wideDecimal(t, 76, 10, true)
	posText := wideDecimalText(76, 10, false)
	negText := wideDecimalText(76, 10, true)

	t.Run("list", func(t *testing.T) {
		query := fmt.Sprintf(`SELECT ['%s', NULL, '%s']::DECIMAL(76,10)[]`, posText, negText)
		var v any
		require.NoError(t, db.QueryRow(query).Scan(&v))
		list, ok := v.([]any)
		require.True(t, ok)
		require.Len(t, list, 3)
		compareDecimal(t, pos, list[0].(Decimal))
		require.Nil(t, list[1])
		compareDecimal(t, neg, list[2].(Decimal))
	})

	t.Run("struct", func(t *testing.T) {
		query := fmt.Sprintf(`SELECT {'a': '%s'::DECIMAL(76,10), 'b': NULL::DECIMAL(76,10), 'c': ['%s'::DECIMAL(76,10)]}`, posText, negText)
		var v any
		require.NoError(t, db.QueryRow(query).Scan(&v))
		m, ok := v.(map[string]any)
		require.True(t, ok)
		compareDecimal(t, pos, m["a"].(Decimal))
		require.Nil(t, m["b"])
		compareDecimal(t, neg, m["c"].([]any)[0].(Decimal))
	})

	t.Run("map", func(t *testing.T) {
		query := fmt.Sprintf(`SELECT MAP {1: '%s'::DECIMAL(76,10)}`, negText)
		var m Map
		require.NoError(t, db.QueryRow(query).Scan(&m))
		compareDecimal(t, neg, m[int32(1)].(Decimal))
	})
}

func TestWideDecimalBind(t *testing.T) {
	db := openDbWrapper(t, ``)
	defer closeDbWrapper(t, db)

	t.Run("value fits 128 bits", func(t *testing.T) {
		// A wide DECIMAL whose value fits into 128 bits keeps its type when bound.
		for _, val := range []Decimal{
			{Width: 39, Scale: 2, Value: big.NewInt(-12345)},
			{Width: 76, Scale: 10, Value: big.NewInt(0)},
			{Width: 255, Scale: 100, Value: new(big.Int).Lsh(big.NewInt(1), 126)},
			{Width: 255, Scale: 0, Value: new(big.Int).Neg(new(big.Int).Lsh(big.NewInt(1), 127))},
		} {
			var typ string
			var got Decimal
			require.NoError(t, db.QueryRow(`SELECT typeof(?), ?`, val, val).Scan(&typ, &got))
			require.Equal(t, fmt.Sprintf("DECIMAL(%d,%d)", val.Width, val.Scale), typ)
			compareDecimal(t, val, got)
		}
	})

	t.Run("value exceeds 128 bits", func(t *testing.T) {
		for _, tt := range []struct{ width, scale int }{{39, 2}, {76, 10}, {255, 0}, {255, 100}, {255, 255}} {
			for _, negative := range []bool{false, true} {
				val := wideDecimal(t, tt.width, tt.scale, negative)
				typ := fmt.Sprintf("DECIMAL(%d,%d)", tt.width, tt.scale)

				// The exact value reaches DuckDB; the cast gives the parameter its type.
				var gotType string
				var got Decimal
				query := fmt.Sprintf(`SELECT typeof(?::%s), ?::%s`, typ, typ)
				require.NoError(t, db.QueryRow(query, val, &val).Scan(&gotType, &got))
				require.Equal(t, typ, gotType)
				compareDecimal(t, val, got)

				// A 39-digit value can still fit into 128 bits and then keeps its type.
				// Otherwise, without a parameter type, the exact text arrives as a VARCHAR.
				wantType := "VARCHAR"
				if decimalFitsHugeInt(val.Value) {
					require.Equal(t, 39, tt.width)
					wantType = typ
				}
				var text string
				require.NoError(t, db.QueryRow(`SELECT typeof(?) || ' ' || ?`, val, val).Scan(&text))
				require.Equal(t, wantType+" "+val.plainString(), text)
			}
		}
	})

	t.Run("insert and compare", func(t *testing.T) {
		_, err := db.Exec(`CREATE TABLE wide_bind (id INTEGER, d DECIMAL(50,3))`)
		require.NoError(t, err)
		small := Decimal{Width: 50, Scale: 3, Value: big.NewInt(-1500)}
		large := wideDecimal(t, 50, 3, true)
		_, err = db.Exec(`INSERT INTO wide_bind VALUES (1, ?), (2, ?), (3, ?)`, small, large, (*Decimal)(nil))
		require.NoError(t, err)

		var got Decimal
		require.NoError(t, db.QueryRow(`SELECT d FROM wide_bind WHERE id = 1`).Scan(&got))
		compareDecimal(t, small, got)
		require.NoError(t, db.QueryRow(`SELECT d FROM wide_bind WHERE id = 2`).Scan(&got))
		compareDecimal(t, large, got)
		var null *Decimal
		require.NoError(t, db.QueryRow(`SELECT d FROM wide_bind WHERE id = 3`).Scan(&null))
		require.Nil(t, null)

		var id int
		require.NoError(t, db.QueryRow(`SELECT id FROM wide_bind WHERE d = ?`, large).Scan(&id))
		require.Equal(t, 2, id)
		require.NoError(t, db.QueryRow(`SELECT id FROM wide_bind WHERE d = ?`, small).Scan(&id))
		require.Equal(t, 1, id)
	})

	t.Run("errors", func(t *testing.T) {
		var got Decimal
		// More digits than the width, beyond 128 bits.
		tooLong := wideDecimal(t, 60, 0, false)
		tooLong.Width = 59
		err := db.QueryRow(`SELECT ?::DECIMAL(59,0)`, tooLong).Scan(&got)
		require.ErrorIs(t, err, errDecimalValueExceedsWidth)
		require.ErrorContains(t, err, "index: 1")

		// More digits than the width, within 128 bits.
		err = db.QueryRow(`SELECT ?`, Decimal{Width: 3, Value: big.NewInt(1000)}).Scan(&got)
		require.ErrorIs(t, err, errDecimalValueExceedsWidth)

		badScale := wideDecimal(t, 60, 0, false)
		badScale.Scale = 61
		err = db.QueryRow(`SELECT ?::DECIMAL(60,0)`, badScale).Scan(&got)
		require.ErrorIs(t, err, errInvalidDecimalScale)
	})
}

func TestWideDecimalAppender(t *testing.T) {
	c, db, conn, a := prepareAppender(t, appenderTypeDefault, `
	CREATE TABLE test (
		id INTEGER,
		d39 DECIMAL(39,2),
		d76 DECIMAL(76,10),
		d255 DECIMAL(255,100),
		l DECIMAL(76,10)[],
		s STRUCT(v DECIMAL(39,2))
	)`)
	defer cleanupAppender(t, c, db, conn, a)

	type row struct {
		d39, d76, d255 *Decimal
	}
	dec := func(d Decimal) *Decimal { return &d }
	rows := []row{
		{dec(wideDecimal(t, 39, 2, false)), dec(wideDecimal(t, 76, 10, false)), dec(wideDecimal(t, 255, 100, false))},
		{dec(wideDecimal(t, 39, 2, true)), dec(wideDecimal(t, 76, 10, true)), dec(wideDecimal(t, 255, 100, true))},
		{nil, nil, nil},
		{
			dec(Decimal{Width: 39, Scale: 2, Value: big.NewInt(0)}),
			dec(Decimal{Width: 76, Scale: 10, Value: big.NewInt(-1)}),
			dec(Decimal{Width: 255, Scale: 100, Value: big.NewInt(255)}),
		},
	}
	value := func(d *Decimal) any {
		if d == nil {
			return nil
		}
		return *d
	}
	for i, r := range rows {
		list := []any{value(r.d76), nil}
		st := map[string]any{"v": value(r.d39)}
		require.NoError(t, a.AppendRow(int32(i), value(r.d39), value(r.d76), value(r.d255), list, st))
	}
	require.NoError(t, a.Flush())

	res, err := db.QueryContext(context.Background(), `SELECT id, d39, d76, d255, l, s FROM test ORDER BY id`)
	require.NoError(t, err)
	defer closeRowsWrapper(t, res)

	compare := func(want *Decimal, got any) {
		if want == nil {
			require.Nil(t, got)
			return
		}
		compareDecimal(t, *want, got.(Decimal))
	}
	i := 0
	for res.Next() {
		var id int
		var d39, d76, d255, l, s any
		require.NoError(t, res.Scan(&id, &d39, &d76, &d255, &l, &s))
		require.Equal(t, i, id)
		compare(rows[i].d39, d39)
		compare(rows[i].d76, d76)
		compare(rows[i].d255, d255)
		list := l.([]any)
		require.Len(t, list, 2)
		compare(rows[i].d76, list[0])
		require.Nil(t, list[1])
		compare(rows[i].d39, s.(map[string]any)["v"])
		i++
	}
	require.NoError(t, res.Err())
	require.Equal(t, len(rows), i)

	// DuckDB reads back the same values.
	var str string
	require.NoError(t, db.QueryRow(`SELECT d255::VARCHAR FROM test WHERE id = 1`).Scan(&str))
	require.Equal(t, wideDecimalText(255, 100, true), str)

	// A value with more digits than the width is rejected.
	tooLong := wideDecimal(t, 40, 2, false)
	err = a.AppendRow(int32(9), tooLong, nil, nil, nil, nil)
	testError(t, err, errAppenderAppendRow.Error(), errDecimalValueExceedsWidth.Error())
}

func TestWideDecimalScalarUDF(t *testing.T) {
	db := openDbWrapper(t, ``)
	defer closeDbWrapper(t, db)
	ctx := context.Background()
	conn := openConnWrapper(t, db, ctx)
	defer closeConnWrapper(t, conn)

	info, err := NewDecimalInfo(76, 10)
	require.NoError(t, err)
	require.NoError(t, RegisterScalarUDF(conn, "wide_negate", &wideDecimalSUDF{info: info}))

	text := wideDecimalText(76, 10, false)
	var typ string
	var got Decimal
	query := fmt.Sprintf(`SELECT typeof(wide_negate(d)), wide_negate(d) FROM (SELECT '%s'::DECIMAL(76,10) AS d)`, text)
	require.NoError(t, conn.QueryRowContext(ctx, query).Scan(&typ, &got))
	require.Equal(t, "DECIMAL(76,10)", typ)
	compareDecimal(t, wideDecimal(t, 76, 10, true), got)

	var null *Decimal
	require.NoError(t, conn.QueryRowContext(ctx, `SELECT wide_negate(NULL::DECIMAL(76,10))`).Scan(&null))
	require.Nil(t, null)
}

func TestWideDecimalTypeInfo(t *testing.T) {
	for _, tt := range []struct{ width, scale uint8 }{{39, 2}, {76, 10}, {255, 0}, {255, 255}} {
		info, err := NewDecimalInfo(tt.width, tt.scale)
		require.NoError(t, err)
		require.Equal(t, TYPE_DECIMAL, info.InternalType())
		details, ok := info.Details().(*DecimalDetails)
		require.True(t, ok)
		require.Equal(t, tt.width, details.Width)
		require.Equal(t, tt.scale, details.Scale)

		lt := info.(*typeInfo).logicalType()
		require.Equal(t, TYPE_BIGNUM, mapping.DecimalInternalType(lt))
		reconstructed, err := newTypeInfoFromLogicalType(lt)
		mapping.DestroyLogicalType(&lt)
		require.NoError(t, err)
		details, ok = reconstructed.Details().(*DecimalDetails)
		require.True(t, ok)
		require.Equal(t, tt.width, details.Width)
		require.Equal(t, tt.scale, details.Scale)
	}

	db := openDbWrapper(t, ``)
	defer closeDbWrapper(t, db)
	ctx := context.Background()
	conn := openConnWrapper(t, db, ctx)
	defer closeConnWrapper(t, conn)

	// The type of a wide DECIMAL result column.
	require.NoError(t, conn.Raw(func(driverConn any) error {
		stmt, err := driverConn.(*Conn).PrepareContext(ctx, `SELECT 1::DECIMAL(38,2) + 1::DECIMAL(38,2)`)
		if err != nil {
			return err
		}
		defer func() { _ = stmt.Close() }()
		info, err := stmt.(*Stmt).ColumnTypeInfo(0)
		if err != nil {
			return err
		}
		details, ok := info.Details().(*DecimalDetails)
		require.True(t, ok)
		require.Equal(t, uint8(39), details.Width)
		require.Equal(t, uint8(2), details.Scale)
		return nil
	}))
}

// The C API reports the width and scale of a DECIMAL as uint8,
// so types wider than 255 digits fail with a specific error.
func TestWideDecimalBeyondCAPI(t *testing.T) {
	db := openDbWrapper(t, ``)
	defer closeDbWrapper(t, db)

	text := wideDecimalText(300, 2, true)
	for _, query := range []string{
		fmt.Sprintf(`SELECT '%s'::DECIMAL(300,2)`, text),
		fmt.Sprintf(`SELECT ['%s'::DECIMAL(300,2)]`, text),
		fmt.Sprintf(`SELECT {'a': '%s'::DECIMAL(300,2)}`, text),
		`SELECT NULL::DECIMAL(256,0)`,
	} {
		var v any
		err := db.QueryRow(query).Scan(&v)
		require.ErrorIs(t, err, errDecimalWidthUnknown)
	}

	// The exact value is available as text.
	var str string
	require.NoError(t, db.QueryRow(fmt.Sprintf(`SELECT '%s'::DECIMAL(300,2)::VARCHAR`, text)).Scan(&str))
	require.Equal(t, text, str)

	// A parameter carries at most 255 digits, and is exact in a wider type.
	val := wideDecimal(t, 255, 2, true)
	require.NoError(t, db.QueryRow(`SELECT ?::DECIMAL(300,2)::VARCHAR`, val).Scan(&str))
	require.Equal(t, wideDecimalText(255, 2, true), str)

	// An appender on such a column fails with the same error.
	c := newConnectorWrapper(t, ``, nil)
	defer closeConnectorWrapper(t, c)
	conn := openDriverConnWrapper(t, c)
	defer closeDriverConnWrapper(t, &conn)
	_, err := conn.(driver.ExecerContext).ExecContext(context.Background(), `CREATE TABLE test (d DECIMAL(300,2))`, nil)
	require.NoError(t, err)
	_, err = NewAppenderFromConn(conn, "", "test")
	require.ErrorIs(t, err, errDecimalWidthUnknown)
}
