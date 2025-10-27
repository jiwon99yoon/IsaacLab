from stl import mesh

def read_stl_simple(file_path):
    """
    numpy-stl 라이브러리를 사용하여 STL 파일 읽기
    
    Parameters:
    file_path (str): STL 파일 경로
    """
    
    # STL 파일 로드
    your_mesh = mesh.Mesh.from_file(file_path)
    
    # 삼각형 개수
    num_triangles = len(your_mesh.vectors)
    
    print(f"STL 파일 경로: {file_path}")
    print(f"삼각형 mesh 개수: {num_triangles}")
    
    return num_triangles


# 사용 예시
if __name__ == "__main__":
    #stl_file_path = "your_file.stl"  # 실제 파일 경로로 변경하세요
    #stl_file_path = "/home/dyros/ws_moveit/src/moveit_resources/panda_description/meshes/collision/link0.stl"
    #stl_file_path = "/home/dyros/ws_moveit/src/hdr_description/meshes/robots/ha006b/visual/base_body.stl"
    #stl_file_path = "/home/dyros/ws_moveit/src/moveit_resources/fanuc_description/meshes/collision/link_1.stl"
    stl_file_path = "/home/dyros/ws_moveit/src/ros2_robotiq_gripper/robotiq_description/meshes/visual/2f_140/robotiq_2f_140_base_link.stl"
    try:
        num_triangles = read_stl_simple(stl_file_path)
    except Exception as e:
        print(f"오류 발생: {e}")
