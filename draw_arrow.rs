// TODO: Figure out macos https://crates.io/crates/cocoa
// TODO: Figure out linux https://crates.io/crates/x11

use windows::{
	Win32::{
		Foundation::*, Graphics::Gdi::*, System::LibraryLoader::*, UI::WindowsAndMessaging::*,
	},
	core::*,
};

struct ArrowData {
	start_xy: (i32, i32),
	end_xy: (i32, i32),
}

pub fn draw(start_xy: (i32, i32), end_xy: (i32, i32)) -> Result<()> {
	unsafe {
		let instance = GetModuleHandleW(None)?;

		let window_class = PCWSTR::from_raw(w!("ArrowOverlayClass").as_ptr());

		let wc = WNDCLASSEXW {
			cbSize: std::mem::size_of::<WNDCLASSEXW>() as u32,
			style: CS_HREDRAW | CS_VREDRAW,
			lpfnWndProc: Some(wndproc),
			hInstance: instance.into(),
			hCursor: LoadCursorW(None, IDC_ARROW)?,
			lpszClassName: window_class,
			..Default::default()
		};

		RegisterClassExW(&wc);

		let arrow_data = Box::new(ArrowData { start_xy, end_xy });
		let arrow_data_ptr = Box::into_raw(arrow_data);

		let hwnd = CreateWindowExW(
			WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_TOPMOST,
			window_class,
			PCWSTR::from_raw(w!("Arrow Overlay").as_ptr()),
			WS_POPUP | WS_VISIBLE,
			0,
			0,
			GetSystemMetrics(SM_CXSCREEN),
			GetSystemMetrics(SM_CYSCREEN),
			None,
			None,
			Some(instance.into()),
			Some(arrow_data_ptr as *mut _),
		)?;

		SetLayeredWindowAttributes(hwnd, COLORREF(0), 0, LWA_COLORKEY)?;

		let mut message = MSG::default();
		while GetMessageW(&mut message, Some(HWND(std::ptr::null_mut())), 0, 0).into() {
			let _ = TranslateMessage(&message);
			DispatchMessageW(&message);
		}
	}

	Ok(())
}

extern "system" fn wndproc(hwnd: HWND, message: u32, wparam: WPARAM, lparam: LPARAM) -> LRESULT {
	unsafe {
		match message {
			WM_CREATE => {
				let create_struct = lparam.0 as *const CREATESTRUCTW;
				if !create_struct.is_null() {
					let arrow_data_ptr = (*create_struct).lpCreateParams;
					SetWindowLongPtrW(hwnd, GWLP_USERDATA, arrow_data_ptr as _);
				}
				LRESULT(0)
			}
			WM_PAINT => {
				let arrow_data_ptr = GetWindowLongPtrW(hwnd, GWLP_USERDATA) as *const ArrowData;
				if !arrow_data_ptr.is_null() {
					let arrow_data = &*arrow_data_ptr;

					let mut ps = PAINTSTRUCT::default();
					let hdc = BeginPaint(hwnd, &mut ps);

					draw_arrow(hdc, arrow_data.start_xy, arrow_data.end_xy);

					EndPaint(hwnd, &ps).expect("Failed to end paint");
				}
				LRESULT(0)
			}
			WM_DESTROY => {
				let arrow_data_ptr = GetWindowLongPtrW(hwnd, GWLP_USERDATA) as *mut ArrowData;
				if !arrow_data_ptr.is_null() {
					std::mem::drop(Box::from_raw(arrow_data_ptr));
					SetWindowLongPtrW(hwnd, GWLP_USERDATA, 0);
				}
				PostQuitMessage(0);
				LRESULT(0)
			}
			_ => DefWindowProcW(hwnd, message, wparam, lparam),
		}
	}
}

fn draw_arrow(hdc: HDC, start_xy: (i32, i32), end_xy: (i32, i32)) {
	let body_pen = unsafe { CreatePen(PS_SOLID, 5, COLORREF(0x00FF00)) };
	let head_pen = unsafe { CreatePen(PS_SOLID, 5, COLORREF(0x00FF00)) };

	let old_pen = unsafe { SelectObject(hdc, GetStockObject(NULL_PEN)) };
	unsafe { SelectObject(hdc, head_pen.into()) };

	let angle = f64::atan2(
		(end_xy.1 - start_xy.1) as f64,
		(end_xy.0 - start_xy.0) as f64,
	);
	let arrow_length = 50.0;
	let arrow_angle = std::f64::consts::PI / 6.0;

	let p1_x = end_xy.0 as f64 - arrow_length * f64::cos(angle - arrow_angle);
	let p1_y = end_xy.1 as f64 - arrow_length * f64::sin(angle - arrow_angle);

	let p2_x = end_xy.0 as f64 - arrow_length * f64::cos(angle + arrow_angle);
	let p2_y = end_xy.1 as f64 - arrow_length * f64::sin(angle + arrow_angle);

	unsafe {
		let _ = MoveToEx(hdc, end_xy.0, end_xy.1, Some(std::ptr::null_mut()));
		let _ = LineTo(hdc, p1_x as i32, p1_y as i32);

		let _ = MoveToEx(hdc, end_xy.0, end_xy.1, Some(std::ptr::null_mut()));
		let _ = LineTo(hdc, p2_x as i32, p2_y as i32);
		SelectObject(hdc, body_pen.into());

		let _ = MoveToEx(hdc, start_xy.0, start_xy.1, Some(std::ptr::null_mut()));
		let _ = LineTo(hdc, end_xy.0, end_xy.1);

		SelectObject(hdc, old_pen);
		let _ = DeleteObject(body_pen.into());
		let _ = DeleteObject(head_pen.into());
	}
}

// let mut stockfish = stockfish::Stockfish::new(
// 	r"C:\Users\root\Downloads\stockfish\stockfish-windows-x86-64-avx2.exe",
// )
// .await?;

// stockfish
// 	.set_position("rnbqkbnr/pp1ppppp/8/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2")
// 	.await?;
// let best_move = stockfish.get_best_move().await?;
// println!("Best move: {:?}", best_move);
